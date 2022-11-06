Layout of data folders:

data
|
|___in-the-wild-images
|		|____test
|   |    |____(empty placeholder)
|   |____train
|   |    |____00000
|   |    |____01000
|   |    |____02000
|   |    |____.....
|   |____test.json
|   |____ffhq-dataset-v2.json (download_required)
|
|___wider_face
    |____train
		|    |____0--Parade
		|    |____1--Handshaking
		|    |____.....
		|____test
		|    |____(Nothing Yet, but this is where it would go)
		|____wider_face_train_bbx_gt.txt
		|____wider_face_test_bbx_gt.txt
		|____wider_face_val_bbx_gt.txt


I will update this data map as required.
This is how FaceDataset.py is expecting the data to be laid out.
