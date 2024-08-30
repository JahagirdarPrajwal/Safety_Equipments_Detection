import os 
import xml.etree.ElementTree as ET
import argparse 

def converter(input_dir,output_dir):##checks for directory and if doesnt exists makes one 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith(".xml"):
            xml_file = os.path.join(input_dir,file)
            tree = ET.parse(xml_file)
            root = tree.getroot()

            image_width = int(root.find('size/width').text)
            image_height = int(root.find('size/height').text)

            yolo_data=[]

            for obj in root.findall('object'):
                class_name = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin= int (bbox.find('xmin').text)
                ymin=int(bbox.find('ymin').text)
                ymax=int(bbox.find('ymax').text)
                xmax=int(bbox.find('xmax').text)


                x_center=(xmin+ xmax ) / 2.0 / image_width
                y_center=(ymin+ymax) / 2.0 / image_height
                width = (xmax - xmin) / image_width
                height = (ymax-ymin ) / image_height

                yolo_data.append(f"0{x_center} {y_center} {width} {height}")

            output_file=os.path.join(output_dir,f"{os.path.splitext(file)[0]}.txt")
            with open(output_file,"w") as f :
                f.write("/n".join(yolo_data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PascalVoc to Yolov8")
    parser.add_argument("input_dir",type=str, help="input directory containing PascalVoc ")
    parser.add_argument("output_dir", type=str, help="output dir where yolov8 will be saved ")

    args = parser.parse_args()
    converter(args.input_dir, args.output_dir)
