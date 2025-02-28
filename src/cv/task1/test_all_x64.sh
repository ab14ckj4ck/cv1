#!/bin/bash

# compare/evaluation script for cv output on 64
# this script is build on imagemagick

# use "sudo apt-get install imagemagick" to install imagemagick
# use "./test_all_x64.sh" to run the script
# use "chmod 755 test_all_x64.sh" for "Permission denied" error

echo ""
echo "imagemagick compare script for cv output on x64"
echo "- comparing output with reference images"
echo "- creates difference images in 'dif/'"
echo "- a 'correct' image will be completely white"
echo "- read comments in this file for more info"
echo ""

# do not pass errors to calling entity
return_error=true
if [[ $* == *--no-error* ]]
then
  echo "Running with flag '--no-error'"
  return_error=false
fi

function label_image {
  image=$1
  text=$2
  convert -background 'rgba(0, 0, 0, .75)' -font DejaVu-Sans -fill white -pointsize 11 \
    label:" $text " -splice 0x3 $image +swap -gravity north \
    -geometry +0+10 -composite $image
}

function diff_images {
  ref_img=$1
  output_img=$2
  diff_img=$3
  ref_size="$(identify -ping -format '%wx%h' $ref_img)"
  output_size="$(identify -ping -format '%wx%h' $output_img)"

  # test if output size is valid (not empty), otherwise corrupt image has been written, so write red template
  if [ ! -z "$output_size" ]; then
    convert $ref_img $output_img -compose difference -composite -negate -contrast-stretch 0 $diff_img
    if [[ $ref_size != $output_size ]]; then
      label_image $diff_img "Different Sizes"
      echo "The image '$output_img' has a different size than the reference."
      return 1
    else
      diff_pixels=$(compare -metric AE $ref_img $output_img null: 2>&1)
      if [ $diff_pixels != "0" ]; then
        echo "There are $diff_pixels different pixels in '$output_img'."
        return 1
      else
        return 0
      fi
    fi
  else
    convert -size 300x300 xc:red $diff_img
    label_image $diff_img "Corrupt Image"
    echo "The image '$output_img' is corrupt."
    return 1
  fi
}

function diff_txt {
  ref_txt=$1
  output_txt=$2

  if cmp -s "$ref_txt" "$output_txt"; then
    return 0  # identical
  else
    echo "Values in $output_txt do not match the reference."
    return 1
  fi
}

if [ -d ./dif ]; then rm -Rf ./dif; fi
mkdir -p dif/


# collect no of different images
declare -i no_diff_images=$no_diff_images+$?
# collect no of missing images
declare -i no_missing_images=$no_missing_images+$?

# collect total no of txt-files in ref
declare -i no_txt=$no_txt+$?
# collect no of different txt-files
declare -i no_diff_txt=$no_diff_txt+$?
# collect no of missing txt-files
declare -i no_missing_txt=$no_missing_txt+$?

# declaration of all testcases
declare -a testcases=(`ls ./data/ref_x64/`)


# ----------------------------------------
# main testcase loop
# ----------------------------------------

for tc in "${testcases[@]}"
do
  echo 'Testing' ${tc} '...'
  mkdir -p dif/${tc}/bonus

  # declaration of all images
  declare -a fnames_png=(`cd ./data/ref_x64/${tc}/ && ls *.png`)
  # declaration of all txt-files (for value outputs), if there are any
  declare -a fnames_txt=(`cd ./data/ref_x64/${tc}/ && ls *.txt 2>/dev/null`)
  if [ ${#fnames_txt[@]} -gt 0 ]; then
    no_txt+=${#fnames_txt[@]}
  fi

  # ----------------------------------------
  # standard task: diff images
  # ----------------------------------------

  for fname in "${fnames_png[@]}"
  do
    # list contains also directories, skip those
    if [ -f data/ref_x64/${tc}/${fname} ]; then
      if [ $fname = "bonus" ]; then
        continue
      fi

      if [ -f output/${tc}/${fname} ]; then
        diff_images data/ref_x64/${tc}/${fname} output/${tc}/${fname} dif/${tc}/${fname}
        no_diff_images+=$?
      else
        convert -size 300x300 xc:red dif/${tc}/${fname}
        no_missing_images+=1
      fi
    fi
  done

  # ----------------------------------------
  # standard task: diff txt-files
  # ----------------------------------------

  for fname in "${fnames_txt[@]}"
  do
    # list contains also directories, skip those
    if [ -f data/ref_x64/${tc}/${fname} ]; then
      if [ $fname = "bonus" ]; then
        continue
      fi

      if [ -f output/${tc}/${fname} ]; then
        diff_txt data/ref_x64/${tc}/${fname} output/${tc}/${fname}
        no_diff_txt+=$?
      else
        no_missing_txt+=1
      fi
    fi
  done

  # ----------------------------------------
  # BONUS task
  # ----------------------------------------

  if [ -d output/${tc}/bonus ]; then

    # declaration of all images
    declare -a fnames_png=(`cd ./data/ref_x64/${tc}/bonus/ && ls *.png`)
    # declaration of all txt-files (for value outputs), if there are any
    declare -a fnames_txt=(`cd ./data/ref_x64/${tc}/bonus/ && ls *.txt 2>/dev/null`)
    if [ ${#fnames_txt[@]} -gt 0 ]; then
      no_txt+=${#fnames_txt[@]}
    fi

    # ----------------------------------------
    # BONUS task: diff images
    # ----------------------------------------

    for fname in "${fnames_png[@]}"
    do
      if [ -f output/${tc}/bonus/${fname} ]; then
        diff_images data/ref_x64/${tc}/bonus/${fname} output/${tc}/bonus/${fname} dif/${tc}/bonus/${fname}
        no_diff_images+=$?
      else
        convert -size 300x300 xc:red dif/${tc}/bonus/${fname}
        label_image dif/${tc}/bonus/${fname} "Missing Image"
        no_missing_images+=1
      fi
    done

    # ----------------------------------------
    # BONUS task: diff txt-files
    # ----------------------------------------

    for fname in "${fnames_txt[@]}"
    do
      if [ -f output/${tc}/bonus/${fname} ]; then
        diff_txt data/ref_x64/${tc}/bonus/${fname} output/${tc}/bonus/${fname}
        no_diff_txt+=$?
      else
        no_missing_txt+=1
      fi
    done
  fi
  echo ${tc}' done.'
done


# ----------------------------------------
# print testcase results
# ----------------------------------------

echo "Number of missing images: $no_missing_images"
echo "Number of different images: $no_diff_images"

# print no of missing/diff txt-files, if there are any in the ref folders
if [ $no_txt -gt 0 ]; then
  echo "Number of missing values (.txt): $no_missing_txt"
  echo "Number of different values (.txt): $no_diff_txt"
fi

# ----------------------------------------
# check testcase SUCCESS / FAILURE
# ----------------------------------------

if [ $no_missing_images -ne 0 ] || [ $no_diff_images -ne 0 ] || [ $no_missing_txt -ne 0 ] || [ $no_diff_txt -ne 0 ] ; then
  echo "At least one ERROR occurred!"
  if [ "$return_error" = true ]; then
    exit 1
  fi
else
  echo "NO errors occurred!"
fi

exit 0