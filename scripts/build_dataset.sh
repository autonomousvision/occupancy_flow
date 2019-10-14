source config.sh

# Make output directories
mkdir -p $BUILD_PATH

mkdir -p $BUILD_PATH/0_points \
        $BUILD_PATH/0_pointcloud \
        $BUILD_PATH/0_meshes #\
        # $BUILD_PATH/0_render

echo " Building Human Dataset for Occupancy Flow Project."
echo " Input Path: $INPUT_PATH"
echo " Build Path: $BUILD_PATH"

echo "Sample points ..."
python sample_mesh.py $INPUT_PATH \
    --n_proc $NPROC --resize \
    --points_folder $BUILD_PATH/0_points \
    --overwrite --float16 --packbits
echo "done!"

echo "Sample pointcloud"
python sample_mesh.py $INPUT_PATH \
    --n_proc $NPROC --resize \
    --pointcloud_folder $BUILD_PATH/0_pointcloud \
    --overwrite --float16
echo "done"

echo "Copy mesh data."
inputs=$(lsfilter $INPUT_PATH)
for m in ${inputs[@]}; do
        m_path="$INPUT_PATH/$m"
        mesh_files=$(lsfilter $m_path) 
        out_path="$BUILD_PATH/0_meshes/$m"
        mkdir -p $out_path
        echo "Copy for model $m"
        for f in ${mesh_files[@]}; do
                mesh_file="$m_path/$f"
                out_file="$out_path/$f"
                cp $mesh_file $out_file
        done
done
echo "done"

# echo "Render sequence (camera on circle)"
# inputs=$(lsfilter $INPUT_PATH $BUILD_PATH/0_render /camera.npz) #
# for f in ${inputs[@]}; do
# # lsfilter $INPUT_PATH $BUILD_PATH/0_render /camera.npz  | parallel -P $NPROC \
#     blender --background --python render_blender.py -- \
#         --output_folder $BUILD_PATH/0_render \
#         --views $N_VIEWS \
#         --camera circle \
#         $INPUT_PATH/$f
# done

