import cv2
import numpy as np

def get_data(file_path='./labels.txt'):
    img_data = {}
    # For background
    found_bg = False

    classes_count = {}
    class_mapping = {}

    with open(file_path, 'r') as f:
        print("---------------------Parsing annotation data---------------------")

        for line in f:
            line_split = line.strip().split(sep=',')
            filename, xmin, ymin, xmax, ymax, label = line_split

            if label not in classes_count:
                classes_count[label] = 1
            else:
                classes_count[label] += 1

            if label not in class_mapping:
                if label == 'bg' and not found_bg:
                    print('Found background class. Will be used for hard negative mining')
                    found_bg = True
                class_mapping[label] = len(class_mapping)

            if filename not in img_data:
                img_data[filename] = {}
                img = cv2.imread(filename)
                rows, cols, _ = img.shape
                img_data[filename]['filepath'] = filename
                img_data[filename]['width'] = cols
                img_data[filename]['height'] = rows
                img_data[filename]['bboxes'] = []

                if np.random.randint(0, 6) > 0:
                    img_data[filename]['imageset'] = 'trainval'
                else:
                    img_data[filename]['imageset'] = 'test'

            img_data[filename]['bboxes'].append({
                'class': label,
                'x1': int(float(xmin)),
                'y1': int(float(ymin)),
                'x2': int(float(xmax)),
                'y2': int(float(ymax))
            })
        all_data = []

        for key in img_data:
            all_data.append(img_data[key])

        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping)- - 1
                class_mapping[key_to_switch] = val_to_switch
        return all_data, classes_count, class_mapping


if __name__ == '__main__':
    get_data()