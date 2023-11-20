
# import OS module
import os
 
# Get the list of all files and directories
image_path = ".\원천데이터\단일경구약제 5000종"
label_path = ".\라벨링데이터\단일경구약제 5000종"

image_list = os.listdir(image_path)
label_list = os.listdir(label_path)
for i in range(len(label_list)):
    label_list[i]= label_list[i].replace('_json', '')

file = open("lisettxt.txt", "w")
file.write("image_list\n")
for image in image_list:
    file.write(image+"\n")
file.write("label_list\n")
for label in label_list:
    file.write(label+"\n")
file.write("none_image\n")
none_image = list(set(label_list) - set(image_list))
for image in none_image:
    file.write(image+"_json OR ")
file.write("none_label\n")
none_label = list(set(image_list) - set(label_list))
for label in none_label:
    file.write(label+" OR ")

file.close()