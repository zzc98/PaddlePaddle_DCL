import argparse
import cv2
import numpy
import paddle.inference as paddle_infer


def standardization(data, mean, sigma):
    return (data - mean) / sigma


def crop(img, top, left, height, width):
    """Crops the given image.
    Args:
        img (np.array): Image to be cropped. (0,0) denotes the top left
            corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
    Returns:
        np.array: Cropped image.
    """

    return img[top:top + height, left:left + width, :]


def center_crop(img, output_size):
    """Crops the given image and resize it to desired size.
        Args:
            img (np.array): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
        Returns:
            np.array: Cropped image.
        """
    h, w = img.shape[0:2]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)


def process(args):
    cv_numpy = cv2.imread(args.img)
    h, w = cv_numpy.shape[:2]
    if min(h, w) == h:
        rh = args.resize_resolution
        rw = int(args.resize_resolution * (w / h))
    else:
        rw = args.resize_resolution
        rh = int(args.resize_resolution * (h / w))
    cv_numpy = cv2.resize(cv_numpy, (rw, rh))
    cv_numpy = center_crop(cv_numpy, (args.crop_resolution, args.crop_resolution))
    cv_numpy = cv2.cvtColor(cv_numpy, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    cv_numpy = cv_numpy / 255
    cv_numpy[0] = standardization(cv_numpy[0], 0.485, 0.229)
    cv_numpy[1] = standardization(cv_numpy[1], 0.456, 0.224)
    cv_numpy[2] = standardization(cv_numpy[2], 0.406, 0.225)
    cv_numpy = numpy.expand_dims(cv_numpy, 0)
    return cv_numpy


def create_predictor(model, params):
    config = paddle_infer.Config(model, params)
    config.disable_gpu()
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    predictor = paddle_infer.create_predictor(config)

    return predictor


def main():
    args = parse_args()

    predictor = create_predictor(args.model_file, args.params_file)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    input_numpy = process(args).astype('float32')
    input_handle.reshape([1, 3, args.crop_resolution, args.crop_resolution])
    input_handle.copy_from_cpu(input_numpy)

    predictor.run()

    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output = output_handle.copy_to_cpu()

    print("prediction: ", output.argmax())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, help="model filename", default='outputs/STATIC/model.pdmodel')
    parser.add_argument("--params_file", type=str, help="parameter filename", default='outputs/STATIC/model.pdiparams')
    parser.add_argument('--img', dest='img', default='resources/Black_Footed_Albatross_0001_796111.jpg', type=str)
    parser.add_argument('--size', dest='resize_resolution', default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution', default=448, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    main()
