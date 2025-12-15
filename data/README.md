# FlyCropper（果蝇裁切器）

简介
-	FlyCropper 是一个用于处理 Label Studio 导出标注（JSON）的工具。它会根据标注中的矩形框（百分比坐标）裁切源图像，并将裁切结果按分类标签保存到不同的文件夹中。

主要特性
-	支持来自多个来源文件夹的图片（例如 `flys/`、`flys2/` 等），直接使用 JSON 中的路径信息查找图片。
-	同一图片可包含多个标注框，都会被单独裁切并保存。
-	跨文件夹处理重名图片时，会为同一基础文件名维护全局递增的裁切序号（例如 `1_crop_1.jpg`, `1_crop_2.jpg`, `1_crop_3.jpg`），保证不覆盖已有裁切结果。
-	按分类标签（例如 `Long`、`Short`）将裁切图像保存到对应子目录下：`cropped_flies/<label>/`。

要求
-	Python 3.8+
-	Pillow（用于图像读取与裁切）

安装依赖
```powershell
python -m pip install --upgrade pip
python -m pip install pillow
```

运行
```powershell
python crop_flies.py
```

脚本位置
-	主脚本：`crop_flies.py`
-	JSON：脚本中默认使用项目根目录下的 `project-1-at-*.json` 文件，你可以在脚本顶部修改 `json_file` 变量以指定其他导出文件。

JSON 要求
-	支持 Label Studio 导出的任务列表格式。脚本会读取每项任务的 `data.image` 字段，该字段通常是相对路径，例如 `flys/MVIMG_20251202_153157.jpg` 或 `flys2/1.jpg`。脚本会把该路径拼接到项目根目录来定位源图片文件。

输出说明
-	所有裁切结果会存放在项目根目录下的 `cropped_flies/` 目录中，按标签分类：
```
cropped_flies/
├─ Long/
│  ├─ 1_crop_1.jpg
│  └─ 1_crop_3.jpg
└─ Short/
   └─ 2_crop_1.jpg
```

命名与重名处理逻辑
-	输出文件名基于源文件的基础名（不含文件夹和扩展名），加上 `_crop_<N>` 后缀，其中 `<N>` 为对该基础名的全局递增序号（跨源文件夹）。例如：
  - `flys/1.jpg` 有两处标注，生成 `1_crop_1.jpg`、`1_crop_2.jpg`。
  - `flys2/1.jpg` 另有一处标注，生成 `1_crop_3.jpg`（序号接着之前的两个）。

注意事项
-	如果 JSON 中引用的图片不存在，脚本会在控制台打印警告并跳过。
-	裁切区域使用 Label Studio 的百分比坐标（`x,y,width,height`），脚本会自动转换为像素并确保坐标在图像范围内。
-	脚本当前默认使用项目根目录下的 JSON 文件名（在 `crop_flies.py` 中配置）。如果你希望通过命令行传参运行，我可以帮你改造脚本来接受 `--json` 与 `--out` 参数。

示例：场景说明
-	假设：
  - `flys/1.jpg` 内标注 2 个目标 → 生成 `cropped_flies/<label>/1_crop_1.jpg`, `1_crop_2.jpg`
  - `flys2/1.jpg` 再标注 1 个目标 → 生成 `cropped_flies/<label>/1_crop_3.jpg`

许可证
-	MIT（你可以根据需要修改或移除许可证）

联系方式
-	如需我把脚本改为支持命令行参数或批量处理多个 JSON（或添加 `requirements.txt`），告诉我我会继续完善。