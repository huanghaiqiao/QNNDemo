set -e
onnx_base_name="best_sim"
onnx_name=${onnx_base_name}.onnx
onnx_path="model/"
output_cpp_path="./build/cpp_path"
output_lib_path="./build/output_lib"
output_context="./build/context"


cd ${QNN_SDK_ROOT}/bin && source ./envsetup.sh && cd -

run_cmd() {
    echo "$@"   # 打印完整命令
    "$@"        # 执行所有参数
}


# qnn-onnx-converter 不能加--input_list量化参考，qnn-context-binary-generator必加config_file


run_cmd ~/qairt/qaisw-v2.37.1.250807093845_124904-auto/bin/x86_64-linux-clang/qnn-onnx-converter \
--input_network ./${onnx_path}${onnx_name} \
--output_path ${output_cpp_path}/${onnx_base_name}.cpp \
--input_dim 'images' 1,3,480,640 \
--preserve_io \
--weight_bw 8 --act_bw 8
# --input_list /home/hq/hq/qnndemo/input_raw/input_list.txt \

run_cmd ~/qairt/qaisw-v2.37.1.250807093845_124904-auto/bin/x86_64-linux-clang/qnn-model-lib-generator \
-c ${output_cpp_path}/${onnx_base_name}.cpp \
-b ${output_cpp_path}/${onnx_base_name}.bin  \
-o ${output_lib_path} \
-t x86_64-linux-clang

# # test_one on linux
# run_cmd ~/qairt/qaisw-v2.37.1.250807093845_124904-auto/bin/x86_64-linux-clang/qnn-net-run \
# --model ${output_lib_path}/x86_64-linux-clang/lib${onnx_base_name}.so \
# --backend ~/qairt/qaisw-v2.37.1.250807093845_124904-auto/lib/x86_64-linux-clang/libQnnHtp.so \
# --input_list /home/hq/hq/qnndemo/input_raw/input_list.txt

~/qairt/qaisw-v2.37.1.250807093845_124904-auto/bin/x86_64-linux-clang/qnn-context-binary-generator \
--backend ~/qairt/qaisw-v2.37.1.250807093845_124904-auto/lib/x86_64-linux-clang/libQnnHtp.so \
--model ${output_lib_path}/x86_64-linux-clang/lib${onnx_base_name}.so \
--binary_file ${onnx_base_name} \
--output_dir ${output_context} \
--config_file /home/hq/hq/qnndemo/x86_64-linux-clang/htp_backend_extensions.json

# # # test_two
run_cmd ~/qairt/2.35.0.250530/bin/x86_64-linux-clang/qnn-net-run \
--retrieve_context ${output_context}/${onnx_base_name}.bin \
--backend ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so \
--input_list /home/hq/hq/qnndemo/input_raw/input_list.txt

# run_cmd ~/qairt/2.35.0.250530/bin/x86_64-linux-clang/qnn-net-run \
# --retrieve_context /home/hq/jiang_test/microphone_a8w8.bin \
# --backend ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so \
# --input_list /home/hq/hq/qnndemo/input_raw/input_list.txt



# run_cmd ~/qairt/qaisw-v2.37.1.250807093845_124904-auto/bin/x86_64-linux-clang/qnn-net-run \
# --model ${output_lib_path}/x86_64-linux-clang/lib${onnx_base_name}.so \
# --input_list /home/hq/hq/qnndemo/input_raw/input_list.txt \
# --backend ${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so \
