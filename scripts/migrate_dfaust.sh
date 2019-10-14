# This script migrates D-FAUST mesh data and the provided point and point cloud data.

IN_PATH=$1 
OUT_PATH=data/Humans/D-FAUST
mesh_folder_name='mesh_seq'

model_files=$(ls $IN_PATH)

echo "Copying mesh data from $IN_PATH to dataset in $OUT_PATH ..."
for m in ${model_files[@]}
do
    echo "Processing model $m ..."
    model_folder_in=$IN_PATH/$m 
    model_folder_out=$OUT_PATH/$m/$mesh_folder_name
    cp -R $model_folder_in $model_folder_out
    echo "done (model)!"
done
echo "done (dataset)!"