Procedure to implement the Material Segmentation to get the cropped region(floor, wall, cieling and rug)


Step-1  
a. Download the weights from the drive and put it in ckpt/ade20k-hrnetv2-c1

link - https://drive.google.com/drive/folders/1Tbi_ym_j5cRPR1RNzSojXDvwKuKAAxMj?usp=sharing

b. to generate the segmentation mask run the following command

python3 -u test.py --imgs $PATH_IMG --gpu $GPU --cfg $CFG

Example - python3 -u test.py --imgs room.jpg --gpu 0 --cfg config/ade20k-hrnetv2.yaml

Output - segmentation mask image with same name but with the extension .png. 

Example - room.png(output image)


Step -2

a. In createMaskJson.py(line no - 90) change the mask image name to the output image name obtained from step-1

Example - mask_image = room.png 

b. Generate the json script by running the following command

python createMaskJson.py

Output = Json script with same name but with the extension .json

Example - room.json


Step-3 

a. In get_cropped_objecy.py(Line no-8:-open the json file obtained as an output from step-2 and 

Line no-9 :- open the original input image .

Example - f = open(room.json) and orig_img = cv2.imread(room.jpg)

b. Generate the cropped regions by running the following command

python get_cropped_object.py

Output - the cropped regions(wall, cieling, rug, floor) if present in the orig image will be saved in the result folder

and this will be the final output.




