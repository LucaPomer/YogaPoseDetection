from scripts.openposeKeypoints import get_openpose_keypoints


for i in range(15):
    base_size = 512
    print(str(i) + 'range '+ str(base_size+(i*16)))
    get_openpose_keypoints((base_size+(i*16)),base_size,'/Users/lucapomer/Documents/bachelor/YogaPoseDetection/accuraccyTest', '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/sceletons')

