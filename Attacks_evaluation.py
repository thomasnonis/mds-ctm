from config import *
from attacks import *
from measurements import *
from transforms import *
from tools import *


images = import_images(IMG_FOLDER_PATH, N_IMAGES_LIMIT, True)


def evaluate_random_attacks(N, img):
    # best_attack = ""
    attack_list = get_random_attacks(N)
    attacked_image = do_attacks(img, attack_list)
    print(attacked_image[1], "WPSNR: ", wpsnr(attacked_image[0], img))
    show_images([(img, "Original")] + [attacked_image], 1, 2)
    return


def evaluate_chosen_attacks(img):
    best_attack = ""
    n_attacks = 10
    attacks_type = ["gaus blur", "avg blur", "jpeg", "sharpen", "awgn", "resize", "median"]
    attacks_array = []
    attacked_images = []

    while int(n_attacks) > 6:
        n_attacks = int(input("How much attacks? "))

    for i in range(n_attacks-1):
        attack = ""
        while attack not in attacks_type:
            attack = input("Which attack? [gaus blur, avg blur, jpeg, sharpen, awgn, resize, median]: ")

        if attack in attacks_array:
            pass
        else:
            attacks_array.append(attack)

    if input("Want to choose parameters? (y or n): ") == "y":
        print(attacks_array)
        for attack in attacks_array:
            print(attack)
            if attack == "gaus blur":
                sigma = input("Insert sigma: ")
                gaussian_blur(img, sigma)

            if attack == "avg blur":
                kernel = input("Insert kernel: ")
                average_blur(img, kernel)

            elif attack == "jpeg":
                qf = input("Insert Quality factor: ")
                jpeg_compression(img, qf)

            elif attack == "sharpen":
                sigma = input("Insert sigma: ")
                alpha = input("Insert alpha: ")
                sharpen(img, sigma, alpha)

            elif attack == "awgn":
                mean = input("Insert mean: ")
                std = input("Insert std: ")
                seed = input("Insert seed: ")
                awgn(img, mean, std, seed)

            elif attack == "resize":
                scale = input("Insert scale: ")
                resize(img, scale)

            elif attack == "median":
                kernel = input("Insert kernel size: ")
                median(img, kernel)
    else:
        attacks_array = get_attacks_list(attacks_array)
        for attack in attacks_array:
            attacked_image = do_attacks(img, [attacks_array[i]])
            attacked_images.append(attacked_image)


    return best_attack

def evaluate_all_attacks(img):
    print("=" * 20)
    best_attack = (0, "none")
    attacks_list = ["GAUS blur", "AVG blur", "jpeg", "sharpen", "AWGN", "Resize", "Median"]
    attacks_list = get_attacks_list(attacks_list)
    print(attacks_list)
    attacked_images = []
    for i in range(6):
        attacked_image = do_attacks(img, [attacks_list[i]])
        attacked_images.append(attacked_image)
        quality = wpsnr(attacked_image[0], img)
        print(attacked_image[1], "WPSNR: ", quality)
        if quality > best_attack[0]:
            best_attack = (quality, attacked_image[1])
    show_images([(img, "Original")] + attacked_images, 2, 4)

    return "Attack with best WPSNR: " + best_attack[1]

def evaluate_single_attack(img):
    attack_types = ["gaus blur", "avg blur", "jpeg", "sharpen", "awgn", "resize", "median"]
    type = input("Which attack do u want to evaluate? [gaus blur, avg blur, jpeg, sharpen, awgn, resize, median]: ")
    if type == "gaus blur":
        rows = 3
        values = 0.5
        attacked_list = []
        for sigma in np.arange(0.1, values, 0.1):
            attacked = gaussian_blur(img, sigma)
            attacked_list.append((attacked, "sigma: " + str(sigma)))
            print("WPSNR (sigma " + str(sigma) + "): ", wpsnr(img, attacked))

        show_images([(img, "Original")] + attacked_list, rows, int(len(attacked_list)//rows)+1)

    if type == "avg blur":
        rows = 2
        values = 15
        attacked_list = []
        for kernel in range(1, values, 2):
            attacked = average_blur(img, kernel)
            attacked_list.append((attacked, "kernel: " + str(kernel)))
            print("WPSNR (kernel " + str(kernel) + "): ", wpsnr(img, attacked))

        show_images([(img, "Original")] + attacked_list, rows, int(len(attacked_list)//rows)+1)

    elif type == "jpeg":
        rows = 1
        values = 20
        attacked_list = []
        for qf in range(5, values, 5):
            attacked = jpeg_compression(img, qf)
            attacked_list.append((attacked, "Quality factor: " + str(qf)))
            print("WPSNR (QF " + str(qf) + "): ", wpsnr(img, attacked))

        show_images([(img, "Original")] + attacked_list, rows, int(len(attacked_list) // rows) + 1)

    elif type == "sharpen":
        rows = 2
        values = 1
        attacked_list = []
        for sigma in np.arange(0.1, values, 0.5):
            for alpha in np.arange(0.1, values, 0.5):
                attacked = sharpen(img, sigma, alpha)
                attacked_list.append((attacked, "Sigma: " + str(sigma) + " Alpha: " + str(alpha)))
        print("WPSNR (Sigma: " + str(sigma) + " Alpha: " + str(alpha) + "): ", wpsnr(img, attacked))

        show_images([(img, "Original")] + attacked_list, rows, int(len(attacked_list) // rows) + 1)


    elif type == "awgn":
        rows = 2
        values = 1
        attacked_list = []
        for mean in range(1, 5):
            for std in np.arange(0.1, values, 0.5):
                seed = randint(0, 1000)
                attacked = awgn(img, mean, std, seed)
                attacked_list.append((attacked, "Mean: " + str(mean) + " Std: " + str(std) + " Seed: " + str(seed)))
        print("WPSNR (Mean: " + str(mean) + " Std: " + str(std) + " Seed: " + str(seed) + "): ", wpsnr(img, attacked))

        show_images([(img, "Original")] + attacked_list, rows, int(len(attacked_list) // rows) + 1)


    elif type == "resize":
        rows = 3
        values = 0.5
        attacked_list = []
        for scale in np.arange(0.1, values, 0.1):
            attacked = resize(img, scale)
            attacked_list.append((attacked, "scale: " + str(scale)))
            print("WPSNR (scale " + str(scale) + "): ", wpsnr(img, attacked))

        show_images([(img, "Original")] + attacked_list, rows, int(len(attacked_list) // rows) + 1)

    elif type == "median":
        rows = 2
        values = 15
        attacked_list = []
        for kernel in range(1, values, 2):
            attacked = median(img, kernel)
            attacked_list.append((attacked, "kernel: " + str(kernel)))
            print("WPSNR (kernel " + str(kernel) + "): ", wpsnr(img, attacked))

        show_images([(img, "Original")] + attacked_list, rows, int(len(attacked_list) // rows) + 1)




def main():
    """for i in images:
        # evaluate_random_attacks(1, i[0])
        print(evaluate_all_attacks(i[0]))"""

    chosen_method = int(input("Choose a method (1 for random attacks; 2 for all attacks; 3 for chosen attacks; 4 evaluate single attack): "))
    if chosen_method == 1:
        for i in images:
            evaluate_random_attacks(1, i[0])
    elif chosen_method == 2:
        print(chosen_method)
        for i in images:
            print(evaluate_all_attacks(i[0]))
    elif chosen_method == 3:
        for i in images:
            evaluate_chosen_attacks(i[0])
    elif chosen_method == 4:
        for i in images:
            evaluate_single_attack(i[0])



if __name__ == "__main__":
    main()