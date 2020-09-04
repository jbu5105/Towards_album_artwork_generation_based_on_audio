from math import ceil
from skimage import io, img_as_ubyte
from skimage.transform import resize
import cv2
from PIL import ImageFile
import os
import json
from shutil import copyfile, move
import numpy as np
from matplotlib import image

from sklearn.neighbors import KDTree
from sklearn import decomposition
from sklearn import preprocessing

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_image(img_dir, new_img_dir_name, x, y):
    """
    Resizes image direction inputted into a new direction

    Inputs:
    :param img_dir: direction of the image you want to resize
    :param new_img_dir_name:  new direction of the resized image XxY resolution
    :param x: X resolution of XxY resolution
    :param y: Y resolution of xxY resolution
    :return: None
    """
    if img_dir:
        img = io.imread(img_dir)
        img_rescaled = resize(img, (x, y, 3), anti_aliasing=True)
        io.imsave(new_img_dir_name, img_as_ubyte(img_rescaled))



def resize_path(path, new_path, x, y):
    """
    Resize all images within a directory

    :param path: directory path
    :param new_path: new directory path
    :param x: X resolution of XxY resolution
    :param y: Y resolution of xxY resolution
    :return: None
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            split = root.replace(os.sep, '/').split('/')[3]
            genre = root.replace(os.sep, '/').split('/')[4]
            #resize_image function declared above
            resize_image(root+'/'+file, new_path+'/'+split+'/'+genre+'/'+file, x, y)



def dhash(image, hashSize=8):
    """
    Hash algorithm function for images. Copied from :
    https://www.pyimagesearch.com/2020/04/20/detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning/

    Inputs:
        :param image: image to hash
        :param hashSize: size of output hash

    :return: hash
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash and return it
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def split_set_genre(path, genre, new_path):
    """
    Split dataset into three splits: train, test and val

    Inputs:
        :param path: initial dataset directory
        :param genre: genre of the dataset directory
        :param new_path: new dataset directory

    :return: None
    """
    # Create split directories if dont exist
    if not os.path.exists(new_path + '/train'):
        os.mkdir(new_path + '/train')
    if not os.path.exists(new_path + '/val'):
        os.mkdir(new_path + '/val')
    if not os.path.exists(new_path + '/test'):
        os.mkdir(new_path + '/test')

    # Create genre directory for each split
    if not os.path.exists(new_path + '/train/' + genre):
        os.mkdir(new_path + '/train/' + genre)
    if not os.path.exists(new_path + '/val/' + genre):
        os.mkdir(new_path + '/val/' + genre)
    if not os.path.exists(new_path + '/test/' + genre):
        os.mkdir(new_path + '/test/' + genre)

    count = 0
    for root, dirs, files in os.walk(path):

        train_len = int(len(files) * 0.8)
        val_len = ceil(len(files) * 0.1)
        test_len = len(files)-train_len-val_len-1

        for file in files:
            count += 1
            if count < train_len:
                copyfile(root + '/' + file, new_path + '/train/' + genre + '/' + file)
            elif train_len <= count < (train_len + val_len):
                copyfile(root + '/' + file, new_path + '/val/' + genre + '/' + file)
            elif count >= train_len + val_len:
                copyfile(root + '/' + file, new_path + '/test/' + genre + '/' + file)



def check_compilation(path_audio, root, genre, file, count_comp, count_new_file, count_repeat):
    '''Check if cover is a compilation by reading tags in json file (path_audio)
    and copies unique and not compilation covers

    Inputs:
        - path_audio: path of the json file with tags
        - root: root of cover file
        - genre: name of genre currently checking
        - file: file name of cover file
        - count_comp: counter of compilations
        - count_new_file: counter of new files addded

    :return number of compilations and new files added
    '''

    with open(path_audio + '/' + file.split('__')[0] + '.json') as json_file:
        data = json.load(json_file)
        check = data['metadata']['tags']
        if 'releasetype' in check:
            album_type = check['releasetype']
            for i in range(len(album_type)):
                if 'compilation' not in album_type[i] and 'Compilation' not in album_type[i]:
                    copy = True
                else:
                    print('Compilation, not copying...', file)
                    copy = False
                    count_comp += 1
                    break
        if 'musicbrainz album type' in check:
            album_type = check['musicbrainz album type']
            for i in range(len(album_type)):
                if 'compilation' not in album_type[i] and 'Compilation' not in album_type[i]:
                    copy = True
                else:
                    print('Compilation, not copying...', file)
                    copy = False
                    count_comp += 1
                    break
        else:
            copy = True

        if copy:
            print("Copying new file")
            count_new_file += 1
            copyfile(root + '/' + file, 'E:/cover_dataset/' + genre + '/' + file + '__' + str(count_repeat) + '.jpg')

    return count_comp, count_new_file



def check_same_covers_hash(path):
    """
    Saves into a dictionary the paths of the images with an specific hash

    Input:
        :param path: dataset directory

    :return: hash dictionary
    """
    hashes = {}
    count = 0
    for root, dir, files in os.walk(path):
        for file in files:
            count += 1
            print(count)
            imagePath = root + '/' + file
            # load the input image and compute the hash
            image = cv2.imread(imagePath)
            #if image is None:
                #os.remove(imagePath)
            #else:
            h = dhash(image)
            # grab all image paths with that hash, add the current image
            # path to it, and store the list back in the hashes dictionary
            p = hashes.get(h, [])
            p.append(imagePath)
            hashes[h] = p

    return hashes



def delete_repeated_covers(hash):
    """
    Deletes cover with the same hash

    Input:
        :param hash: hash dictionary extracted with the previous function

    :return: None
    """
    for key in hash.keys():
        if len(hash[key]) > 1:
            for i, file in enumerate(hash[key],1):
                os.remove(hash[key][i-1])
                if i == (len(hash[key])-1):
                    break



def json_to_npy(json_path, new_npy_path):
    """
    Converts json files from the lowlevel dumps of AcousticBrainz into numpy arrays

    Inputs:
        :param json_path: json file path
        :param new_npy_path: new numpy array path

    :return: None
    """
    keys = ['lowlevel','rhythm','tonal']
    l = []
    with open(json_path) as f:
        data = json.load(f)
        for key in keys:
            for item, key_1 in data[key].items():
                if type(key_1) == float or type(key_1) == int:
                    l.append(key_1)

                elif type(key_1) == list:
                     mean = np.mean(key_1)
                     std = np.std(key_1)
                     l.append(mean)
                     l.append(std)


                elif type(key_1) == dict:
                    for k, key_2 in key_1.items():
                        if type(key_2) == list:
                            mean = np.mean(key_2)
                            std = np.std(key_2)
                            l.append(mean)
                            l.append(std)
                        else:
                            if item == 'spectral_spread' and (k == 'dvar2' or k == 'dvar' or k == 'var'):
                                break
                            elif item == 'spectral_rolloff' and (k == 'dvar2' or k == 'dvar' or k == 'var'):
                                break
                            elif item == 'spectral_centroid' and (k == 'dvar2' or k == 'dvar' or k == 'var'):
                                break
                            else:
                                l.append(key_2)


    n = np.array(l)
    np.save(new_npy_path, n)




def concat_4x4_images(path):
    """
    Concatenates 4x4 images into the same file

    Input:
        :param path: directory of covers dataset

    :return: array with all images and a list with the paths for all images
    """
    count = 0
    images = np.empty((1, 48))
    dir_images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            count += 1
            print(count)
            new_image = image.imread(root + '/' + file)
            new_image = new_image.reshape(1, -1)
            images = np.concatenate((images, new_image), axis=0)
            dir_images.append(root + '/' + file)

    return images[1:], dir_images



def similar_4x4_images(images, dir_images, threshold):
    """
    Moves similar images into the same split

    Inputs:
        :param images: array of 4x4 images calculated with the previous function
        :param dir_images: a list with the paths for all images extracted with the previous function
        :param threshold: threshold which selects the distance in the KDTree to decide if split in same dataset or not

    :return: None
    """
    count = 0
    c = 0
    tree = KDTree(images, metric='manhattan')
    for i in range(images.shape[0]):
        c += 1
        print(c)

        dist, ind = tree.query(images[i:i + 1], k=10)

        for count, d in enumerate(dist[0]):
            if 0 < count < (len(dist[0]) - 1):
                interval = dist[0][count + 1] - d
                if interval > 150:
                    num_similar_int = count + 1
                    break
                else:
                    num_similar_int = 0

        num_similar_thr = len([x for x in dist[0] if x < threshold])

        num_similar = min(num_similar_int, num_similar_thr)

        if 1 < num_similar:
            ind_split = ind.tolist()[0][0]
            split = dir_images[ind_split].split('\\')[1]
            current_file = 'E:/dataset/cover' + dir_images[ind_split].split('E:/dataset/cover_4')[1]
            if os.path.exists(current_file):
                for j in range(num_similar):
                    if j > 0:
                        index = ind.tolist()[0][j]
                        dir_4x4 = dir_images[index]
                        new_dir = 'E:/dataset/cover' + '\\' + split + '\\' + dir_4x4.split('\\')[2]
                        di = 'E:/dataset/cover' + dir_4x4.split('E:/dataset/cover_4')[1]
                        if os.path.exists(di):
                            move(di, new_dir)



def PCA(path, new_path):
    """
    Calculates PCA for arrays contained in a certain directory

    Inputs:
        :param path: directory of arrays
        :param new_path: new directory to save compressed arrays

    :return: None
    """
    file_names = []
    genres = []
    splits = []
    c = 0
    tot = np.zeros((1,2048))
    for root, dirs, files in os.walk(path):
        for file in files:
            x = np.load(root+'/'+file)
            x = x.reshape(1,-1)
            tot = np.concatenate((tot,x), axis=0)

            genre = root.replace(os.sep, '/').split('/')[11]
            split = root.replace(os.sep, '/').split('/')[12]

            file_names.append(file)
            genres.append(genre)
            splits.append(split)
            print(tot.shape)

    tot = tot[1:]

    pca = decomposition.PCA(n_components=580)
    pca.fit(tot)
    X = pca.transform(tot)

    min_max_scaler = preprocessing.MinMaxScaler((0, 1))
    min_max_scaler.fit(X)
    t = min_max_scaler.transform(X)

    for i in range(X.shape[0]):
        np.save(new_path+'/'+ splits[i] + '/' + genres[i] + '/' + file_names[i], X[i])
