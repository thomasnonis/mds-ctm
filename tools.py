import os
import numpy as np
import cv2
import pickle
from transforms import *
from config import *
import matplotlib.pyplot as plt
import random
import concurrent.futures
import traceback


def wpsnr_to_mark(wpsnr: float) -> int:
    """Convert WPSNR to a competition mark

    Args:
        wpsnr (float): the WPSNR value in dB

    Returns:
        int: The mark that corresponds to the WPSNR value according to the competition rules
    """
    if wpsnr >= 35 and wpsnr < 50:
        return 1
    if wpsnr >= 50 and wpsnr < 54:
        return 2
    if wpsnr >= 54 and wpsnr < 58:
        return 3
    if wpsnr >= 58 and wpsnr < 62:
        return 4
    if wpsnr >= 62 and wpsnr < 66:
        return 5
    if wpsnr >= 66:
        return 6
    return 0


def generate_watermark(size_h: int, size_v: int = 0, save: bool = False) -> np.ndarray:
    """Generates a random watermark of size (size_h, size_v)

    Generates a random watermark of size (size_h, size_v) if size_v is specified,
    otherwise generates a square watermark of size (size_h, size_h)

    Args:
        size_h (int): Horizontal size
        size_v (int, optional): Vertical size. Defaults to size_h.

    Returns:
        np.ndarray: Random watermark of the desired size
    """
    if size_v == 0:
        size_v = size_h

    # Generate a watermark
    mark = np.random.uniform(0.0, 1.0, size_v * size_h)
    mark = np.uint8(np.rint(mark))
    if save is True:
        np.save('mark.npy', mark)
    return mark.reshape((size_v, size_h))


def show_images(list_of_images: list, rows: int, columns: int, show: bool = True) -> None:
    """Plot a list of images in a grid of size (rows, columns)

    The list of images must be a list of tuples (image, title), such as:
    [(watermarked, "Watermarked"), (attacked, "Attacked"), ...]

    Args:
        list_of_images (list): List of (image: list, title: str) tuples
        rows (int): number of rows in the grid
        columns (int): number of columns in the grid
        show (bool, optional): Whether to plt.show() the images within the function or let the user plt.show() at a different time. Defaults to True.
    """
    for (i, (image, label)) in enumerate(list_of_images):
        plt.subplot(rows, columns, i + 1)
        plt.title(list_of_images[i][1])
        plt.imshow(list_of_images[i][0], cmap='gray')

    if show is True:
        plt.show()


def embed_into_svd(img: np.ndarray, watermark: list, alpha: float) -> tuple:
    """Embeds the watermark into the S component of the SVD decomposition of the image

    Args:
        img (np.ndarray): Image in which to embed the watermark
        watermark (list): Watermark to embed
        alpha (float): Embedding strength coefficient

    Returns:
        tuple: (Watermarked image: np.ndarray, SVD key matrices: tuple)
    """
    (svd_u, svd_s, svd_v) = np.linalg.svd(img)

    # Convert S from a 1D vector to a 2D diagonal matrix
    svd_s = np.diag(svd_s)

    # Embed the watermark in the SVD matrix
    for x in range(0, watermark.shape[0]):
        for y in range(0, watermark.shape[1]):
            svd_s[x][y] += alpha * watermark[x][y]

    (svd_s_u, svd_s_s, svd_s_v) = np.linalg.svd(svd_s)

    # Convert S from a 1D vector to a 2D diagonal matrix
    svd_s_s = np.diag(svd_s_s)

    # Recompose matrices from SVD decomposition
    watermarked = svd_u @ svd_s_s @ svd_v
    # key = svd_s_u @ svd_s @ svd_s_v

    return (watermarked, (svd_s_u, svd_s_v))


def extract_from_svd(original_img, watermarked_img, svd_key, alpha):
    # Perform SVD decomposition of original_img
    svd_o_u, svd_o_s, svd_o_v = np.linalg.svd(original_img)

    # Convert S from a 1D vector to a 2D diagonal matrix
    svd_o_s = np.diag(svd_o_s)

    # Perform SVD decomposition of watermarked_img
    svd_w_u, svd_w_s, svd_w_v = np.linalg.svd(watermarked_img)

    # Convert S from a 1D vector to a 2D diagonal matrix
    svd_w_s = np.diag(svd_w_s)

    # Reconstruct S component using embedding key components
    s_ll_d = svd_key[0] @ svd_w_s @ svd_key[1]

    # Initialize the watermark matrix
    watermark = np.zeros([MARK_SIZE, MARK_SIZE], dtype=np.float64)

    # Extract the watermark
    for i in range(0, MARK_SIZE):
        for j in range(0, MARK_SIZE):
            watermark[i][j] = (s_ll_d[i][j] - svd_o_s[i][j]) / alpha

    return watermark


def save_parameters(img_name: str, svd_key: tuple) -> None:
    """Saves the necessary parameters for the detection into parameters/<img_name>_parameters.txt

    Args:
        img_name (str): Name of the image
        svd_key (tuple): Tuple containing the SVD key matrices for the reverse algorithm
    """
    if not os.path.isdir('parameters/'):
        os.mkdir('parameters/')
    f = open('parameters/' + img_name + '_parameters.txt', 'wb')
    pickle.dump((img_name, svd_key), f, protocol=2)
    f.close()


def read_parameters(img_name: str) -> tuple:
    """Retrieves the necessary parameters for the detection from parameters/<img_name>_parameters.txt

    Args:
        img_name (str): Name of the image

    Returns:
        tuple: (Name of the image: str, Embedding strength coefficient: float, SVD key matrices for the reverse algorithm: np.ndarray)
    """
    # print("IMGNAME: ", img_name)
    f = open('parameters/' + img_name + '_parameters.txt', 'rb')
    (img_name, svd_key) = pickle.load(f)
    f.close()
    return img_name, svd_key


def import_images(img_folder_path: str, num_images: int, shuffle: bool = False) -> list:
    """Loads a list of all images contained in a folder and returns a list of (image, name) tuples
    Args:
        img_folder_path (str): Relative path to the folder containing the images (e.g. 'images/')
    Returns:
        list: List of (image, name) tuples
    """
    if not os.path.isdir(img_folder_path):
        exit('Error: Images folder not found')

    images = []
    paths = os.listdir(img_folder_path)
    if shuffle:
        random.shuffle(paths)
    for img_filename in paths[:num_images]:
        # (image, name)
        images.append((cv2.imread(img_folder_path + img_filename, cv2.IMREAD_GRAYSCALE), img_filename.split('.')[-2]))

    print('Loaded', num_images, 'image' + ('s' if num_images > 1 else ''))

    return images


def extract_watermark(original_img: np.ndarray, img_name: str, watermarked_img: np.ndarray, alpha: int, level: int,
                      subbands: list) -> np.ndarray:
    """Extracts the watermark from a watermarked image by appling the reversed embedding algorithm,
    provided that the proper configuration file and the original, unwatermarked, image are available.

    Args:
        original_img (np.ndarray): Original unwatermarked image
        img_name (str): Name of the image
        watermarked_img (np.ndarray): Image from which to extract the watermark
        subbands (list): List of subbands where to extract the watermark

    Returns:
        np.ndarray: Extracted watermark
    """
    original_coeffs = wavedec2d(original_img, level)
    watermarked_coeffs = wavedec2d(watermarked_img, level)
    watermarks = []
    for subband in subbands:
        original_band = None
        watermarked_band = None
        if subband == "LL":
            original_band = original_coeffs[0]
            watermarked_band = watermarked_coeffs[0]
        elif subband == "HL":
            original_band = original_coeffs[1][0]
            watermarked_band = watermarked_coeffs[1][0]
        elif subband == "LH":
            original_band = original_coeffs[1][1]
            watermarked_band = watermarked_coeffs[1][1]
        elif subband == "HH":
            original_band = original_coeffs[1][2]
            watermarked_band = watermarked_coeffs[1][2]
        else:
            raise Exception(f"Subband {subband} does not exist")

        (_, svd_key) = read_parameters(img_name + '_' + str(alpha) + '_' + subband + str(level))
        watermark = extract_from_svd(original_band, watermarked_band, svd_key, alpha)
        watermarks.append(watermark)

    final_watermark = np.zeros([MARK_SIZE, MARK_SIZE], dtype=np.float64)

    for watermark in watermarks:
        final_watermark += watermark
    final_watermark = final_watermark / len(subbands)

    return final_watermark


# Split function
def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def embed_watermark(original_img: np.ndarray, img_name: str, watermark: np.ndarray, alpha: float, level,
                    subbands: list) -> np.ndarray:
    """Embeds a watermark into the S component of the SVD decomposition of an image's LL DWT subband

    Args:
        original_img (np.ndarray): Image in which to embed the watermark
        img_name (str): Name of the image
        watermark (np.ndarray): Watermark to embed
        alpha (float): Watermark embedding strength coefficient
        subbands (list): List of subbands where to embed the watermark

    Returns:
        np.ndarray: Watermarked image
    """
    coeffs = wavedec2d(original_img, level)

    for subband in subbands:
        band = None
        if subband == "LL":
            band = coeffs[0]
        elif subband == "HL":
            band = coeffs[1][0]
        elif subband == "LH":
            band = coeffs[1][1]
        elif subband == "HH":
            band = coeffs[1][2]
        else:
            raise Exception(f"Subband {subband} does not exist")

        band_svd, svd_key = embed_into_svd(band, watermark, alpha)
        save_parameters(img_name + '_' + str(alpha) + '_' + subband + str(level), svd_key)

        if subband == "LL":
            coeffs[0] = band_svd
        elif subband == "HL":
            coeffs[1] = (band_svd, coeffs[1][1], coeffs[1][2])
        elif subband == "LH":
            coeffs[1] = (coeffs[1][0], band_svd, coeffs[1][2])
            band = coeffs[1][1]
        elif subband == "HH":
            coeffs[1] = (coeffs[1][0], coeffs[1][1], band_svd)
        else:
            raise Exception(f"Subband {subband} does not exist")

    return waverec2d(coeffs)


def make_dwt_image(img_coeffs: list) -> np.ndarray:
    """Creates a DWT image from a given set of DWT coefficients

    Args:
        img (np.ndarray): DWT coefficients

    Returns:
        np.ndarray: DWT image
    """
    levels = len(img_coeffs) - 1
    original_size = img_coeffs[0].shape[0] * (2 ** levels)
    img = np.zeros((original_size, original_size), dtype=np.float64)
    size = 0
    i = levels
    for level in range(1, levels + 1):
        size = int(original_size / (2 ** level))
        img[size:size * 2, 0:size] = img_coeffs[i][0]
        img[0:size, size:size * 2] = img_coeffs[i][1]
        img[size:size * 2, size:size * 2] = img_coeffs[i][2]
        i -= 1

    size = int(original_size / (2 ** levels))
    img[0:size, 0:size] = img_coeffs[0]

    return img


def embed_watermark_tn(original_img: np.ndarray, img_name: str, watermark: np.ndarray, alpha: float,
                       beta: float) -> np.ndarray:
    from measurements import nvf, csf
    coeffs = wavedec2d(original_img, DWT_LEVEL)
    h1 = coeffs[2][0]
    h2 = coeffs[1][0]
    v1 = coeffs[2][1]

    watermarked_h2, svd_key = embed_into_svd(h2, watermark, alpha)
    h1_strength = nvf(csf(h1), 75, 3)
    v1_strength = nvf(csf(v1), 75, 3)

    # [0,1] to [-1,1]
    watermark = (2 * watermark) - 1

    for x in range(0, h1_strength.shape[0]):
        for y in range(0, h1_strength.shape[1]):
            h1[x][y] += (1 - h1_strength[x][y]) * watermark[x % MARK_SIZE][y % MARK_SIZE] * beta
            v1[x][y] += (1 - v1_strength[x][y]) * watermark[x % MARK_SIZE][y % MARK_SIZE] * beta
    save_parameters(img_name + '_' + str(alpha) + '_' + str(beta), svd_key)

    coeffs[2] = (h1, v1, coeffs[2][2])
    coeffs[1] = (watermarked_h2, coeffs[1][1], coeffs[1][2])
    return waverec2d(coeffs)


def extract_watermark_tn(original_img: np.ndarray, img_name: str, watermarked_img: np.ndarray, alpha: float,
                         beta: float) -> np.ndarray:
    """Extracts the watermark from a watermarked image by appling the reversed embedding algorithm,
    provided that the proper configuration file and the original, unwatermarked, image are available.

    Args:
        original_img (np.ndarray): Original unwatermarked image
        img_name (str): Name of the image
        watermarked_img (np.ndarray): Image from which to extract the watermark

    Returns:
        np.ndarray: Extracted watermark
    """
    from measurements import nvf, csf

    (_, svd_key) = read_parameters(img_name + '_' + str(alpha) + '_' + str(beta))

    original_coeffs = wavedec2d(original_img, DWT_LEVEL)
    original_h1 = original_coeffs[2][0]
    original_h2 = original_coeffs[1][0]
    original_v1 = original_coeffs[2][1]

    attacked_coeffs = wavedec2d(watermarked_img, DWT_LEVEL)
    attacked_h1 = attacked_coeffs[2][0]
    attacked_h2 = attacked_coeffs[1][0]
    attacked_v1 = attacked_coeffs[2][1]

    attacked_watermarks = np.empty((1, MARK_SIZE, MARK_SIZE))

    # This is of size 128x128, while all others are 256x256!
    attacked_watermarks[0] = extract_from_svd(original_h2, attacked_h2, svd_key, alpha)

    original_h1_strength = nvf(csf(original_h1), 75, 3)
    original_v1_strength = nvf(csf(original_v1), 75, 3)

    # Should be (attacked_h1 - original_h1), but the watermark comes out flipped this way, so swap numerators
    # h1_new = h1_old + (1-h1_strength[x][y]) * watermark[x % MARK_SIZE][y % MARK_SIZE] * beta
    # (h1_new - h1_old) / ((1-h1_strength[x][y]) * beta)
    attacked_watermark_h1 = (original_h1 - attacked_h1) / ((1 - original_h1_strength) * beta)
    attacked_watermark_v1 = (original_v1 - attacked_v1) / ((1 - original_v1_strength) * beta)

    # [-1,1] to [0,1]
    attacked_watermark_h1 = np.interp(attacked_watermark_h1, (attacked_watermark_h1.min(), attacked_watermark_h1.max()),
                                      (0, 1))
    attacked_watermark_v1 = np.interp(attacked_watermark_v1, (attacked_watermark_v1.min(), attacked_watermark_v1.max()),
                                      (0, 1))

    # show_images([(attacked_watermarks[0], 'Extracted SVD'),(attacked_watermark_h1, 'attacked_watermark_h1'), (attacked_watermark_v1, 'attacked_watermark_v1')],1,3)

    # Split the 256x256 watermark we embedded into MARK_SIZExMARK_SIZE subwatermarks anc calculate mean watermark on each axis
    attacked_watermark_h1 = np.mean(split(attacked_watermark_h1, MARK_SIZE, MARK_SIZE), axis=0)
    attacked_watermark_v1 = np.mean(split(attacked_watermark_v1, MARK_SIZE, MARK_SIZE), axis=0)

    # show_images([(attacked_watermark_h1, 'MEAN attacked_watermark_h1'), (attacked_watermark_v1, 'MEAN attacked_watermark_v1')],1,2)
    attacked_watermarks = np.append(attacked_watermarks, [attacked_watermark_h1], axis=0)
    attacked_watermarks = np.append(attacked_watermarks, [attacked_watermark_v1], axis=0)

    # Calculate the mean of watermarks extracted from SVD, H1 and V1
    return np.mean(attacked_watermarks, axis=0)


def embed_watermark_dct(original_img: np.ndarray, img_name: str, watermark: np.ndarray, alpha: float, level, subbands: list) -> np.ndarray:
    """Embeds a watermark into the S component of the SVD decomposition of an image's LL DWT subband

    Args:
        original_img (np.ndarray): Image in which to embed the watermark
        img_name (str): Name of the image
        watermark (np.ndarray): Watermark to embed
        alpha (float): Watermark embedding strength coefficient
        subbands (list): List of subbands where to embed the watermark

    Returns:
        np.ndarray: Watermarked image
    """
    coeffs = wavedec2d(original_img, level)

    for subband in subbands:
        band = None
        if subband == "LL":
            band = coeffs[0]
        elif subband == "HL":
            band = coeffs[1][0]
        elif subband == "LH":
            band = coeffs[1][1]
        elif subband == "HH":
            band = coeffs[1][2]
        else:
            raise Exception(f"Subband {subband} does not exist")

        # print(band)

        band = dct(dct(band, axis=0, norm='ortho'), axis=1, norm='ortho')

        band_svd, svd_key = embed_into_svd(band, watermark, alpha)
        save_parameters(img_name + '_' + subband + str(level), svd_key)

        band_svd = idct(idct(band_svd, axis=1, norm='ortho'), axis=0, norm='ortho')

        if subband == "LL":
            coeffs[0] = band_svd
        elif subband == "HL":
            coeffs[1] = (band_svd, coeffs[1][1], coeffs[1][2])
        elif subband == "LH":
            coeffs[1] = (coeffs[1][0], band_svd, coeffs[1][2])
        elif subband == "HH":
            coeffs[1] = (coeffs[1][0], coeffs[1][1], band_svd)
        else:
            raise Exception(f"Subband {subband} does not exist")

    """watermark = waverec2d(coeffs)
    print(wpsnr(watermark, original_img))
    plt.figure()
    plt.subplot(121)
    plt.title("Original")
    plt.imshow(original_img, cmap='gray')
    plt.subplot(122)
    plt.title("Watermarked")
    plt.imshow(watermark, cmap='gray')
    plt.show()"""
    return waverec2d(coeffs)


def extract_watermark_dct(original_img: np.ndarray, img_name: str, watermarked_img: np.ndarray, alpha: int, level: int, subbands: list) -> np.ndarray:
    """Extracts the watermark from a watermarked image by appling the reversed embedding algorithm,
    provided that the proper configuration file and the original, unwatermarked, image are available.

    Args:
        original_img (np.ndarray): Original unwatermarked image
        img_name (str): Name of the image
        watermarked_img (np.ndarray): Image from which to extract the watermark
        subbands (list): List of subbands where to extract the watermark

    Returns:
        np.ndarray: Extracted watermark
    """
    original_coeffs = wavedec2d(original_img, level)
    watermarked_coeffs = wavedec2d(watermarked_img, level)
    watermarks = []
    for subband in subbands:
        original_band = None
        watermarked_band = None
        if subband == "LL":
            original_band = original_coeffs[0]
            watermarked_band = dct(dct(watermarked_coeffs[0], axis=0, norm='ortho'), axis=1, norm='ortho')
        elif subband == "HL":
            original_band = original_coeffs[1][0]
            watermarked_band = dct(dct(watermarked_coeffs[1][0], axis=0, norm='ortho'), axis=1, norm='ortho')
        elif subband == "LH":
            original_band = original_coeffs[1][1]
            watermarked_band = dct(dct(watermarked_coeffs[1][1], axis=0, norm='ortho'), axis=1, norm='ortho')
        elif subband == "HH":
            original_band = original_coeffs[1][2]
            watermarked_band = dct(dct(watermarked_coeffs[1][2], axis=0, norm='ortho'), axis=1, norm='ortho')
        else:
            raise Exception(f"Subband {subband} does not exist")

        original_band_u, original_band_s, original_band_v = np.linalg.svd(original_band)
        original_band_s = np.diag(original_band_s)

        watermarked_band_u, watermarked_band_s, watermarked_band_v = np.linalg.svd(watermarked_band)
        watermarked_band_s = np.diag(watermarked_band_s)

        (_, svd_key) = read_parameters(img_name + '_' + subband + str(level))
        # original_s_ll_d_u, original_s_ll_d_s, original_s_ll_d_v
        s_band_d = svd_key[0] @ watermarked_band_s @ svd_key[1]

        # Initialize the watermark matrix
        watermark = np.zeros([MARK_SIZE, MARK_SIZE], dtype=np.float64)

        # Extract the watermark

        for i in range(0, MARK_SIZE):
            for j in range(0, MARK_SIZE):
                watermark[i][j] = (s_band_d[i][j] - original_band_s[i][j]) / alpha
        watermarks.append((watermark, subband + " " + img_name))

    final_watermark = np.zeros([MARK_SIZE, MARK_SIZE], dtype=np.float64)

    for watermark in watermarks:
        final_watermark += watermark[0]
    final_watermark = final_watermark / len(subbands)

    #show_images(watermarks + [(final_watermark, "Final")], 1, 3)
    """plt.figure()
    plt.subplot(121)
    plt.title("Final Watermark")
    plt.imshow(final_watermark, cmap='gray')
    plt.show()"""
    return final_watermark


def save_model(scores: list, labels: list, threshold: float, tpr: float, fpr: float, new_params) -> None:
    """Saves the model trained models/model_<alpha>_<level>_<subband>.txt
    The scores and label are saved too in case we want to continue training

    Args:
        scores (list): Scores list
        labels (list): Labels list
        threshold (float): The threshold
        tpr (float): The true positive rate
        fpr (float): The false positive rate
        alpha (float): The alpha used for embedding
        level (int): The level used for embedding
        subband (list): The subband(s) used for embedding
    """
    directory = 'models/'
    if not os.path.isdir(directory):
        os.mkdir(directory)
    params = []
    for x in new_params:
        if type(x) == list:
            params.append('-'.join(x))
        else:
            params.append(str(x))

    params = '_'.join(params)
    f = open(directory + 'model_' + params, 'wb')
    pickle.dump((scores, labels, threshold, tpr, fpr, new_params), f, protocol=2)
    f.close()


def read_model(name: str) -> None:
    """Loads a model from a file

    Args:
        name (str): Name of the model to be loaded
    """
    f = open('models/model_' + name, 'rb')
    values = list(pickle.load(f))

    (scores, labels, threshold, tpr, fpr, params) = (values[0], values[1], values[2], values[3], values[4], values[5])
    f.close()
    return scores, labels, threshold, tpr, fpr, params


def exists_model(name: str) -> None:
    """Checks if a model exists

    Args:
        name (str): Name of the model to be checked
    """
    return os.path.exists('models/model_' + name)


def create_model(params, order_of_execution):
    from measurements import compute_thr_multiple_images
    watermarked_images = []
    images = params[0]
    params = params[1:]
    embedding_function = params[0]
    extraction_function = params[1]
    attacks = params[-2]
    show_threshold = params[-1]
    watermark = params[2]
    new_params = ()
    for original_img, img_name in images:
        watermarked_img = None
        if embedding_function == embed_watermark:
            alpha = params[3]
            level = params[4]
            subband = params[5]
            new_params = (alpha, level, subband)  # Doing this in a loop is useless, is needed only once
            watermarked_img = embed_watermark(original_img, img_name, watermark, alpha, level, subband)
        elif embedding_function == embed_watermark_tn:
            alpha = params[3]
            beta = params[4]
            new_params = (alpha, beta)  # Doing this in a loop is useless, is needed only once
            watermarked_img = embed_watermark_tn(original_img, img_name, watermark, alpha, beta)
        elif embedding_function == embed_watermark_dct:
            alpha = params[3]
            level = params[4]
            subband = params[5]
            new_params = (alpha, level, subband)  # Doing this in a loop is useless, is needed only once
            watermarked_img = embed_watermark_dct(original_img, img_name, watermark, alpha, level, subband)
        else:
            print(f'Embedding function {embedding_function} does not exist!')
        watermarked_images.append((original_img, watermarked_img, img_name))

    scores, labels, (threshold, tpr, fpr) = compute_thr_multiple_images(extraction_function, watermarked_images,
                                                                        watermark, new_params, attacks, show_threshold)
    save_model(scores, labels, threshold, tpr, fpr, new_params)
    return order_of_execution, threshold, tpr, fpr, new_params


def multiprocessed_workload(function, work):
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # future_to_report = {executor.submit(function, *unit_of_work, order_of_execution): unit_of_work for order_of_execution,unit_of_work in enumerate(work)}
        future_to_report = {executor.submit(function, unit_of_work, order_of_execution): unit_of_work for
                            order_of_execution, unit_of_work in enumerate(work)}

    tmp_results = []
    for future in concurrent.futures.as_completed(future_to_report):
        result = future_to_report[future]
        try:
            r = future.result()
            tmp_results.append(r)
        except Exception as exc:
            print("Exception!", "{}".format('%r generated an exception: %s' % (result, traceback.format_exc())))
    tmp_results = sorted(tmp_results, key=lambda x: x[0])
    results = [result[1:] for result in tmp_results]

    return results
