source config.sh

# Function for processing a single model
organize_model() {
  filename=$(basename -- $3)
  modelname="${filename%.*}"
  output_path="$2/$modelname"
  build_path=$1

  points_folder="$build_path/0_points/$modelname"
  points_out_folder="$output_path/points_seq"

  # img_folder="$build_path/0_render/$modelname"
  # img_out_folder="$output_path/img"

  pointcloud_folder="$build_path/0_pointcloud/$modelname"
  pointcloud_out_folder="$output_path/pcl_seq"

  mesh_folder="$build_path/0_meshes/$modelname"
  mesh_out_folder="$output_path/mesh_seq"
  
  # if [ -d $points_folder ] \
  #   && [ -d $pointcloud_folder ] \
  #   && [ -d $img_folder ] \
  #   && [ -d $mesh_folder ] \
  if [ -d $points_folder ] \
    && [ -d $pointcloud_folder ] \
    && [ -d $mesh_folder ] \
    ; then
    echo "Copying model $output_path"
    mkdir -p "$output_path"

    cp -rT $points_folder $points_out_folder
    cp -rT $pointcloud_folder $pointcloud_out_folder
    #Qcp -rT $img_folder $img_out_folder
    cp -rT $mesh_folder $mesh_out_folder
  fi
}

echo "Installing Humans Dataset for Occupancy Flow"
echo "Output Directory: $OUTPUT_PATH"

export -f organize_model

# Make output directories
mkdir -p $OUTPUT_PATH

# Run install
ls $INPUT_PATH | parallel -P $NPROC \
  organize_model $BUILD_PATH $OUTPUT_PATH {}

# Copy Split Files
cp split_files/* $OUTPUT_PATH/

echo "done!"