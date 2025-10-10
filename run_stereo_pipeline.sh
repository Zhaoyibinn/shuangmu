#!/bin/bash
# 一键执行双目立体矫正、基线计算与FoundationStereo预处理
# 使用指定的图像和标定文件路径

set -e  # 遇到错误立即退出

# ========= 配置路径 =========
LEFT="/home/root123/gongcheng/shuangmu/whx_biaoding/rosL1.png"
RIGHT="/home/root123/gongcheng/shuangmu/whx_biaoding/rosR1.png"
EXTRI="/home/root123/gongcheng/shuangmu/biaoding/extrinsics_d435_20250915.yml"
INTRI="/home/root123/gongcheng/shuangmu/biaoding/intrinsics_d435_20250915.yml"

# FoundationStereo配置 (按需修改)
FS_REPO="$HOME/src/FoundationStereo"
CKPT="$FS_REPO/pretrained_models/23-51-11/model_best_bp2.pth"

# 工作目录
WORK_DIR="/home/root123/gongcheng/shuangmu"
OUTPUT_DIR="$WORK_DIR/stereo_output"

echo "=========================================="
echo "双目立体矫正与基线计算 - 一键处理脚本"
echo "=========================================="

# 切换到工作目录
cd "$WORK_DIR"

# 检查输入文件
echo "检查输入文件..."
for file in "$LEFT" "$RIGHT" "$INTRI" "$EXTRI"; do
    if [ ! -f "$file" ]; then
        echo "错误: 文件不存在: $file"
        exit 1
    fi
    echo "✓ $file"
done

echo
echo "========= 1) 执行双目立体矫正与基线计算 ========="
python3 one_click_stereo.py \
    --left "$LEFT" \
    --right "$RIGHT" \
    --intri "$INTRI" \
    --extr "$EXTRI" \
    --out_dir "$OUTPUT_DIR" \
    --alpha 0.0

echo
echo "========= 2) 显示处理结果 ========="
if [ -f "$OUTPUT_DIR/stereo_report.txt" ]; then
    echo "=== 基线计算报告 ==="
    cat "$OUTPUT_DIR/stereo_report.txt"
else
    echo "报告文件未生成"
fi

echo
echo "========= 3) 检查输出文件 ========="
echo "输出目录: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR/" 2>/dev/null || echo "输出目录为空"

# 显示intrinsics.txt内容
if [ -f "$OUTPUT_DIR/intrinsics.txt" ]; then
    echo
    echo "=== FoundationStereo内参文件 ==="
    echo "文件: $OUTPUT_DIR/intrinsics.txt"
    echo "内容:"
    cat "$OUTPUT_DIR/intrinsics.txt"
fi

echo
echo "========= 4) (可选) 运行FoundationStereo ========="
if [ -f "$CKPT" ] && [ -d "$FS_REPO" ]; then
    echo "发现FoundationStereo环境，准备运行推理..."

    # 激活conda环境 (可选)
    # conda activate foundation_stereo 2>/dev/null || true

    # 运行FoundationStereo
    python3 "$FS_REPO/scripts/run_demo.py" \
        --left_file "$OUTPUT_DIR/left_rect.png" \
        --right_file "$OUTPUT_DIR/right_rect.png" \
        --ckpt_dir "$CKPT" \
        --out_dir "$OUTPUT_DIR/foundation_stereo_out" \
        --intrinsics_file "$OUTPUT_DIR/intrinsics.txt" \
        --hiera 1

    echo "FoundationStereo输出保存在: $OUTPUT_DIR/foundation_stereo_out"

else
    echo "跳过FoundationStereo (权重或代码库未找到)"
    echo "  权重路径: $CKPT"
    echo "  代码库路径: $FS_REPO"
    echo "  若要运行FoundationStereo，请:"
    echo "  1. 安装FoundationStereo到 $FS_REPO"
    echo "  2. 下载权重到 $CKPT"
    echo "  3. 重新运行此脚本"
fi

echo
echo "=========================================="
echo "处理完成! 主要输出文件:"
echo "  矫正后左图: $OUTPUT_DIR/left_rect.png"
echo "  矫正后右图: $OUTPUT_DIR/right_rect.png"
echo "  极线验证图: $OUTPUT_DIR/combined_with_epipolar_lines.png"
echo "  FoundationStereo内参: $OUTPUT_DIR/intrinsics.txt"
echo "  详细报告: $OUTPUT_DIR/stereo_report.txt"
echo "=========================================="