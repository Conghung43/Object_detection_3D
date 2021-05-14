# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
# import pyzed.sl as sl
import time
import numpy as np
from datetime import datetime
# from test_dataset import get_object_dicts


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        # self.metadata = MetadataCatalog.get(
        #     cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        # )


        # a = get_object_dicts("box/train")
        d = "train"
        # DatasetCatalog.register("box_" + d, lambda d=d: get_object_dicts("box/" + d))
        DatasetCatalog.register("box", self.fake_func)

        MetadataCatalog.get("box_" + d).thing_classes = ['box']
        self.metadata = MetadataCatalog.get("box")


        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def fake_func(self):
        return {}

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, zed, image, runtime_parameters):
        while True:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame_gen = image.get_data()
                frame_gen = cv2.cvtColor(frame_gen, cv2.COLOR_RGB2BGR)
                yield frame_gen
            else:
                break

    def _depth_from_video(self, zed, depth, runtime_parameters):
        while True:
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                depth_gen = depth.get_data()
                yield depth_gen
            else:
                break

    def _frame_depth_from_video(self, pipeline, pc):
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_map = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            points = pc.calculate(depth_frame)
            v = points.get_vertices()
            point_cloud = np.asanyarray(v).view(np.float32).reshape(480, 640, 3) 

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)

            yield [color_image, depth_map, point_cloud]



    def run_on_video(self, pipeline, pc):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, point_cloud, depth, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('image_raw.jpg', frame)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)

                if predictions.has("pred_masks"):
                    masks = predictions.pred_masks
                
                frame_visualizer = Visualizer(frame, self.metadata)
                mask_layer = frame_visualizer.get_mask_layer(masks=masks)
                # vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)

                # depth_layer = []
                # start_time = time.time()
                # for i in range(len(mask_layer)):
                #     mask = mask_layer[i].mask
                #     for y in range(len(mask)):
                #         concate_depth = mask[y]*depth[y]
                #         # concate_depth = np.setdiff1d(concate_depth,np.array([float('nan')]))
                #         # concate_depth = np.nan_to_num(concate_depth)
                #         depth_layer.append(concate_depth)
                #     # f =  open('dummy_data/depth_map_{}.npy'.format(datetime.now().second), 'wb') 
                #     f =  open('dummy_data/depth_map.npy', 'wb') 
                #     np.save(f, depth_layer)
                #     np.save(f, point_cloud)
                #     np.save(f, mask)
                #     f.close()
                # end_time = time.time()
                # print('elapse time = ', end_time - start_time)
                # print('depth_layer ready ')

            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # import numpy as np
            # import matplotlib.pyplot as plt

            # ax = plt.axes(projection='3d')

            # ax.scatter3D(np.array(point_cloud_layer)[:,0], np.array(point_cloud_layer)[:,1], np.array(point_cloud_layer)[:,2], cmap='Greens', s=0.5)
            # plt.show()

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = vis_frame.get_image()
            # vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            cv2.imwrite('dummy_data/image_seg.jpg', vis_frame)
            return vis_frame

        # frame_gen = self._frame_from_video(zed, image, runtime_parameters)
        # depth_gen = self._depth_from_video(zed, depth, runtime_parameters)
        data_gen = self._frame_depth_from_video(pipeline, pc)
        
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame, depth, point_cloud in data_gen:
                # if cam_data.dtype == 'uint8':
                #     frame = cam_data
                # else:
                #     depth = cam_data
                    
                yield process_predictions(frame, point_cloud, depth, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
