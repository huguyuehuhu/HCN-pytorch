links_to_download=("http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s02.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s03.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s07.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s01.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s03.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s06.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s07.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s02.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s04.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s05.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s06.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s02.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s03.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s06.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s05s02.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s05s03.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s02.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s03.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s04.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s07s01.zip"
	           "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s07s03.zip")

save_in_directory="../SBU-Kinect-Interaction/"

#create the directory to save the dataset
mkdir -p ${save_in_directory}

for link in "${links_to_download[@]}";do
    wget -P ${save_in_directory} $link
done

#unzip all zip files
unzip -o ${save_in_directory}\*.zip -d ${save_in_directory}

#remove zip files after unzip
rm ${save_in_directory}/*.zip 

