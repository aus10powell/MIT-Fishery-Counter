"""
Modified from Nils Tijtgat's script, found at: https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/

Purpose: 
    Generates train.txt and test.txt files, which contain the paths to training/test images and their annotations.

Modifications: 
    In its current state, this script has been altered to add all images in the given folder to a single train.txt
file, as opposed to a certain percentage being set aside for testing. This was done in order to specifically select a test
set that contained representative numbers of each class, rather than a random set of images that may not sufficient numbers
of all classes. If a random selection is desired, uncomment the appropriate lines.
Directory Constraints: Run from the directory containing both the image files and txt files.

Usage: 
    python process.py ../data/rv_boxed_herring/Images ../
"""


def main():
    import glob, os, sys

    # Current directory
    images_dir = sys.argv[1]
    print(images_dir)

    # Directory where the data will reside.
    # if len(sys.argv) < 2:
    #     text_path_data = os.getcwd()
    # else:
    text_files_dir = sys.argv[2]
    print(text_files_dir)
    # Percentage of images to be used for the test set. Uncomment if desired.
    # percentage_test = 10;

    # Create and/or truncate train.txt and, if desired, test.txt.
    file_train = open("train.txt", "w")
    # file_test = open('test.txt', 'w')

    # Populate train.txt and test.txt, if desired.
    counter = 1
    # index_test = round(100 / percentage_test)
    # Change the extension to appropriate image type. Note, however, that, although training works with other image types,
    # certain functions (recall, valid) only work with .jpg files. If these functions are desired, you must convert the images.

    jpeg_files = list(glob.iglob(os.path.join(images_dir, "*.jpg")))
    print("Num files found in jpeg_files: " + str(len(jpeg_files)))
    print(jpeg_files)
    for pathAndFilename in jpeg_files:
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        # Uncomment if populating test.txt
        # if counter == index_test:
        # counter = 1
        # file_test.write(path_data + title + '.jpg' + "\n")
        # else:
        file_train.write(
            text_files_dir + title + ".jpg" + "\n"
        )  # re-indent if populating test.txt
        # counter = counter + 1
    file_train.close()


if __name__ == "__main__":
    main()
