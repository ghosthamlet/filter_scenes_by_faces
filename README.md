
# Filter video scenes by faces

just tested on ubuntu with python3.6.6


## Install
install python3.6.6 ffmpeg3.0.2

git clone https://github.com/ghosthamlet/filter_scenes_by_faces.git

cd filter_scenes_by_faces/

pip install -r requirements.txt


## Use
put videos in data/videos/

put person photos in data/persons/

one photo has to contain just one face, more photos with different face angle, more accuracy the result video

then run: 

python filter_scenes_by_faces.py

the filtered video will in data/output/

