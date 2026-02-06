RK系列开发板部署yolov5
Windows安装ubuntu子系统
首先在电脑上安装Linux子系统，通过wsl去安装。
ubuntu系统（Linux系统）适合开发。
直接安装ubuntu为主系统，或者使用双系统模式的话，如果是英伟达的显卡（大部分都是），训练深度学习模型需要用到英伟达显卡驱动，在ubuntu系统中安装这个很麻烦，一不小心就会让系统直接崩溃。
所以子系统就成为了一个很好的选择，不仅可以无条件使用主机的资源，还不用像平行双系统那样麻烦的切换系统，特别方便
首先在电脑搜索栏输入功能，打开勾选 适用于windows的Linux子系统 和 虚拟机平台 
1.	打开windows命令提示符，输入 wsl.exe --update  安装wsl。
2.	打开微软商店，搜索ubuntu22.04 安装ubuntu。
3.	打开windows命令提示符，输入 wsl -l  检查子系统有无成功安装，显示具体系统版本信息，说明安装成功。
4.	默认子系统存储在C盘，如果C盘空间不足，可以转移到其他盘符，在D盘创建wsl文件夹，打开windows 命令提示符，输入 wsl --export Ubuntu-22.04 D:\wsl\Ubuntu.tar 将C盘的ubuntu系统备份到D盘中，然后输入 wsl --unregister Ubuntu-22.04 删除C盘的ubuntu，继续输入 wsl --import Ubuntu-22.04 D:\wsl D:\wsl\Ubuntu.tar 从D盘恢复子系统，结束后输入wsl -l 检查子系统信息是否存在。
5.	注意，该ubuntu子系统无法使用 su root 打开root用户，必须要在windows命令提示符内输入ubuntu2204 config --default-user root 来去手动切换普通用户和管理员，切换回普通用户只需将上述输入内容中的root改为普通用户名即可




英伟达显卡驱动安装
在命令行输入nvidia-smi，有以下输出说明有驱动，这一步骤跳过即可，否则没有需要手动安装显卡驱动。
 
原生Ubuntu系中安装驱动：http://t.csdnimg.cn/Pt9tC

安装Anaconda 
关于其解释，我在本文档第20页紫色字体有说明！！
在我们的子系统Ubuntu下，新建空闲文件夹下输入指令wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh ，wget下载安装包
 
完成之后可以看见路径下已经有安装包了
 
赋予脚本执行权限并且执行
 
接下来一路回车即可
   

 
再次回到命令行后就安装完成了
 
验证是否安装
 
如果命令行输出的是conda：command not found,说明没有将conda添加到环境路径上，编辑~/.bashrc文件
 
在最后一行加上

要注意这里不能直接复制，要和自己的安装路径对的上 
闭坑指南：按照上面操作还是 conda: command not found时，在重新打开一个终端输入命令就会有反应。

输入conda activate，进入基本环境
 
注意，输入后可能conda init错误，输入 source activate 重新进入虚拟环境，输入conda deactivate 退出虚拟环境，就可以正常使用conda activate 来打开虚拟环境了。
至此，conda安装完毕。
YOLOV5源码获取
打开官方yolov5在github上的地址并下载https://github.com/ultralytics/yolov5    推荐下载yolov5-7.0版本，有些版本可能下载无法运行，所以到github之后一定要找版本选择，选择7.0版本，本文档同目录下我已经下载好了yolov5的各版本官方源码，直接用就行
 
 

下载后将该yolov5源码保存在ubuntu系统下你指定的文件并且解压，如下图
 

环境安装
打开我们安装的ubuntu子系统还可以通过在vscode中安装wsl插件，然后在vscode中打开该子系统，所以这极大方便了我们以后去书写代码。
当vscode中的wsl插件打开并且检测到Ubuntu系统后（如下图） 
就说明vscode可以链接我们电脑当中的ubuntu子系统了，然后在vscode打开解压后的yolo文件夹。
 
打开目录中的requirements.txt，这其中包含了库名称，我们可以通过pip一键安装，这就是为什么我在安装环境之前先获取代码
 
命令行输入conda create -n yolov5 python==3.11创建conda环境，python版本可以不一样，要在3.8以上
 
输入y
 
安装完成，输入conda avtivate yolov5进入环境（后续进行的操作都要在现在创建的这个环境里面进行）
 
一定要先进入环境，输入pip install -r requirements.txt，通过该文本安装环境，注意不要加sudo，否则不会装在conda环境中
 
如果下载太慢，或者报错TimeoutError，大概率是因为从国外源下载，速度太慢下载超时了，输入pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple换成清华源（换源也可以直接
在pip install -r requirements.txt的后面加上 
-i https://pypi.tuna.tsinghua.edu.cn/simple

也就是终端输入：
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

可以指定你要在哪个网站去下载，比直接将系统下载源换成清华源好用一些。





 
等待一段时间，下载完成
 
我们还需要验证以下torch版本是GPU版本还是CPU版本，在命令行输入python,引用包并查看版本,如果都类似以下的输出（xxx+cuxxx），说明安装的是GPU版本，即cuda版本
 
import torch, torchvision
torch.__version__, torchvision.__version__
如果返回以下结果，说明默认安装了cpu版本的torch
 
注意！请先检查你的电脑是何种显卡：
1.如果是集成显卡，那么你可以安装cuda的版本的pytorch但无法使用gpu加速训练过程，所以装哪个版本都没问题
2.如果是独立显卡，应该安装cuda版torch，如果默认安装了cpu版（如上情况），需要依照以下步骤重新安装
首先回到命令行，输入pip uninstall torch torchvision，卸载当前的torch，提示是否删除输入Y
 
输入nvidia-smi，记住你系统的cuda版本支持
 
进入pytorch官网：https://pytorch.org （国外网站，有时候可能需要多等待一会）
下滑，到以下部分，Ubuntu系统选择Linux，直接在windows中操作的选择Windows，选择一个CUDA版本（需要保证此处选择的版本<=刚刚指令查看系统的cuda版本），其他选项不变，复制下方命令
 
粘贴到命令行，开始下载，等待下载完成，下载爆红，很多时候都是网络不稳定，下载断开后继续下载，重复执行，直至下载结束！！！！
 
下载完成后，再次验证版本，方法和前文一致
报错篇：
出现 `torch.load` 函数加载模型权重，由于 `weights_only=True` 的限制，导致无法加载某些全局变量（如 `models.yolo.Model`）报错时。这个问题通常与 PyTorch 的安全机制有关，特别是在 PyTorch 2.6 及以上版本中，`torch.load` 默认启用了 `weights_only=True`，以防止加载不受信任的代码。
解决方案
2. **禁用 `weights_only=False`
   如果你确定权重文件是安全的，可以直接将 `weights_only` 参数设置为 `False`：
   在models内的common.py文件中找到
model = torch.jit.load(w, _extra_files=extra_files, map_location=device)    将其修改为
model = torch.jit.load(w, _extra_files=extra_files, map_location=device, weights_only=False)```
5. **降级 PyTorch 版本**：
   如果上述方法都无法解决问题，可以考虑降级 PyTorch 到 2.6 之前的版本（如 2.5 或更低版本），因为这些版本默认不启用 `weights_only=True`。
   pip install torch==2.5.0  # 降级 Torch
同时为了保持torch和torch_vision兼容性，也需要将torch_vision降级
pip install torch_vision==0.20.0  # 降级 Torch_vision
### 总结
这个错误通常是由于 PyTorch 的安全机制导致的，特别是在加载模型权重时。你可以通过禁用 `weights_only` 安全检查或降级 PyTorch 版本来解决这个问题。











PS：为了验证所有包都部署到位，你可以再次输入pip install -r requirements.txt，如果全屏无报错，就说明OK了
 
至此，你已经完成了所有环境的准备，并且所有的python第三方库都安装在了刚刚创建的conda环境中，可以轻松通过conda工具与其他环境进行切换
接下来我们试下能不能进行目标检测，首先我们通过vscode进入Ubuntu子系统，然后进入yolov5目录下，在vscode终端中使用source activate激活虚拟环境，再使用conda activate yolov5 进入yolov5虚拟环境中，如下图。
 

然后我们在vscode中编译运行yolov5文件目录下的detect.py文件，也可以在终端中使用python detect.py 来去编译运行该文件，当然可能会出现import torch导入错误报错信息，我们在其后面加上#type:ignore即可消除该错误，运行后该文件会链接https://github.com/ultralytics/yolov5 来去下载官方的物体检测.pt模型，如下图
 
由于在该步骤下载速度缓慢，所以我们可以直接访问上图网站，去下载图片中所示任意满足我们要求的官方.pt权重文件，下载好后直接放入yolov5目录下，如下图 

然后继续按上面的步骤去运行detect.py文件，这次yolov5会读取存放在其目录下data中image文件内的图片，利用官方训练好的.pt物体检测权重模型来去生成检测结果到runs/detect目录下，我们只需要导入我们需要检测的物体图片到data/image目录下，运行detect.py文件即可得到结果，注意注意！！！！！！！最终得到检测结果，那么证明我们的yolov5已经成功部署，但是我们不可能只用官方的权重文件，我们需要自己去训练属于自己的模型并且模型转化，部署到开发板，那么继续往下看！！！！！！！


模型训练
首先获取数据集，所谓数据集，就是你需要让yolov5去识别的东西的图片集，比如做一个人脸识别的项目，那么你需要采集你全方位的脸部照片（越多越好）然后去放在yolov5中让其训练识别，所以
第一步：，获取你要训练的图片（由于我们最后是在开发板上去识别和处理数据，而且摄像头也是在开发板上调用的，那么我们直接在开发板上调用摄像头去拍摄批量图片比我们拿手机拍摄图片要更接近我们真实的检测情况）
在与本文档同目录下存放有我写好的批量拍摄多张图片的代码，直接放在开发板上略微修改就可以跑。

第二步：1、我们获取到批量的图片后，要按照一定格式去存放这些图片，方便我们导入到yolov5源码中。
新建datasets文件夹，在该文件夹下再新建两个文件夹，images和labels，images里面新建两个文件夹train和val，labels里面新建两个文件夹train和val。（将批量得到的图片的90%放入images/train目录下，10%放入images/val目录下）
2、首先解释下images文件夹下train和val这两个文件夹，train存放我们需要进行训练的图片，val存放验证图片（我们训练完毕后得到的模型与val里的图片进行验证），所以比如我们刚才批量获取到的图片是1000张，那么我们需要分出来大概900张存放到train文件下用来训练，剩下100张全部放在val文件下用来最后的验证。
3、再来解释下labels文件夹下train和val这两个文件夹，train存放我们需要进行训练的图片的标注信息（后续会讲到），val存放验证图片的标注信息（我们训练完毕后得到的模型与val里的图片标注信息进行验证），所以我们通过标注软件labellmg将images/train目录下的图片标注后以YOLO格式（text后缀）存放到labels/train目录下，将images/val目录下的图片标注后以YOLO格式（text后缀）存放到labels/val目录下。当然这一步是后续的步骤，现在跳过即可。
第三步：打开标注软件labellmg，该软件可以网上下载，为了方便，直接在本文档同目录下打开即可（如遇到打不开的情况，将该文件夹放到我们的ubuntu子系统内再去打开），在打开之前需要注意的是，比如我们要标注人脸识别，那么我们需要在该软件目录打开data文件内的文档，将你需要标注的人脸名称类别添加进去，例如下图








添加好后我们再打开该软件，设置为YOLO格式输出，如下图 
然后再点击左侧第二个选项打开我们在上面创建的datasets/images/train，再点击左侧第三个选项设置输出标注信息保存到datasets/labels/train下，记住，val文件下的图片也都是同样操作，然后就可以开始标注了，左上角打开View，选择Auto save mode即可实现自动保存，键盘A和D是切换上下图片，按一下W，按住鼠标左键即可将目标框起来，框好后选择标签即可成功标注，由于开启了自动保存，所以直接按D切换下一张继续标注。

第四步：标注完成后，检查datasets/labels/train和datasets/labels/val 两个目录下的标注文件是否完整并且将labellmg生成的class.txt文件放在datasets内，没问题便直接将datasets文件夹放入yolov5源码目录下，如下图

然后打开yolov5目录下data文件夹，新建myvoc.yaml文件，该文件用以让yolov5中的train.py找到我们的数据集在哪里。myvoc.yaml文件内容如下图 
Path是我们的datasets的存放位置，由于是相对路径，datasets在myvoc.yaml的上一级目录，所以我们直接写 ./datasets   train和val在这里是在path目录下的也就是datasets目录下的，所以我们直接写images/train以及image/val，
然后再看底下的names，CLASSESS类别，也就是我们的标签名或者说标签类别，你在labellmg软件的data数据文档内的类别对应填上去即可，例如下图 
注意，这里一定要按照顺序，labellmg内首位标签名是谁，这里第0个就是谁，而且后续要与标签信息YOLO格式输出text文档内的首序号对应检查，如果这里填写的标签名和你在data下的文档内的首序号不一致，会报错或者训练数据异常，例如下
 
这个是标注后《某个类别生成的yolo格式文档》，你的yaml中CLASS类别0处要将类别填写为上一行红圈标出的序号0所对应的类别名称，一定要在CLASS填写正确，不然报错。

然后再打开yolov5目录下models/yolov5s.yaml文件，将nc也就是类别数改成你的类别数量，也就是你的标签名的个数。这里也要与上述对应，如果上面myvoc.yaml内的names内的类别个数不等于nc的数目，也是会出现报错和训练数据异常。
然后在yolov5s.yaml下有anchors锚点信息，这个不要修改，因为yolo会自动计算最有锚点，所以使用默认即可，不然会导致后续开发板识别时检测框格式异常。
到后期我们在开发板端的推理代码中，post-process（后处理）函数中的锚点anchors的具体参数数值应当与电脑端训练时yolov5s.yaml中的锚点参数保持一致，如果不一致，会造成检测框忽大忽小
在 YOLOv5 中，**默认的锚点（anchors）是基于 COCO 数据集预计算的**，但官方代码在训练时会自动根据你的数据集重新计算更合适的锚点。以下是具体分析：
### **1. 是否需要手动设置？**
- **默认锚点**：对于大多数通用场景（尤其是目标尺寸分布与 COCO 数据集相似时），**直接使用默认锚点即可**。YOLOv5 的默认锚点已针对常见目标进行了优化。
- **自动计算**：YOLOv5 在训练时默认启用 `autoanchor` 功能，**自动对数据集的标注框进行 k-means 聚类**，生成更匹配当前数据集的锚点。这是推荐的方式，无需手动干预。
- **手动修改**：只有以下情况需要手动调整锚点：
  - 目标尺寸分布与 COCO 差异极大（例如极端长宽比或极小/极大目标）。
  - 自动计算的锚点效果不佳（可通过验证集指标判断）。
### **2. 如何操作？**
#### **(1) 使用默认锚点**
- 保持 `yolov5s.yaml` 中的默认锚点不变：
  ```yaml
  anchors:
    - [10,13, 16,30, 33,23]   # P3/8
    - [30,61, 62,45, 59,119]  # P4/16
    - [116,90, 156,198, 373,326]  # P5/32
#### **(2) 启用自动锚点计算**
- 在训练命令中添加 `--autoanchor` 参数（YOLOv5 默认已启用，无需显式添加）：
  ```bash
  python train.py --autoanchor
  训练时会输出类似以下信息，表示锚点与数据集的匹配度：
  AutoAnchor: thr=0.25: 0.9992 best possible recall, 4.97 anchors past thr
#### **(3) 手动优化锚点**
- **步骤 1**：使用 YOLOv5 提供的 `utils/autoanchor.py` 单独计算锚点：
  ```bash
  python utils/autoanchor.py --data your_dataset.yaml
  脚本会输出针对你的数据集优化的锚点值。
- **步骤 2**：将新锚点更新到 `yolov5s.yaml` 中，并关闭自动锚点计算：
  ```yaml
  anchors:
    - [new_p3_anchors]
    - [new_p4_anchors]
    - [new_p5_anchors]
  python train.py --noautoanchor
### **3. 验证锚点效果**
- **关键指标**：关注训练日志中的 `BPR` (Best Possible Recall) 值。若 `BPR > 0.98`，说明锚点与数据集匹配良好；若 `BPR < 0.95`，建议重新计算锚点。
### **总结建议**
- **优先使用默认锚点 + 自动计算**：YOLOv5 的 `autoanchor` 机制已足够鲁棒。
- **特殊数据集**：若目标尺寸分布极端，手动计算并固定锚点可能提升性能（但需通过实验验证）。

接下来打开train.py文件中的parse_opt函数，指定模型配置文件默认--cfg  default=’ ’,我这里不清楚到底是不是默认打开的是yolov5s.yaml,所以我索性给强制加进去了，也就是default = ROOT / ‘models/yolov5s.yaml’，上一行的模型初始权重是yolov5s.pt<-用初始权重文件来训练我们自己的模型。
再继续--data 改为自己的数据集地址文件myvoc.yaml,  默认的好像是官方识别物体种类的coco128数据集地址文件，直接改即可。改完后如下图
 
然后替换激活函数，进入models/common.py文件内，将conv类中的default_act = nn.SiLU()替换为default_act = nn.ReLU(), RK3588的NPU对ReLU函数的支持更优化，计算效率高，能够提高npu占用率和推理速度。

到目前为止我们的训练准备工作结束了，然后进入 yolov5的conda虚拟环境下python train.py 训练我们的模型，模型训练好后会在runs/train/exp/weights目录下，我们就会得到一个best.pt模型，在exp目录下还有该模型的性能分析和结果图片，我们可以去观察下，如果模型验证图片检测异常（标签名不对应，识别错误等），则需要返回继续检查上述步骤。


模型转换	

Pt转onnx：
我们将训练好的best.pt文件复制粘贴到yolov5目录下，在目录下打开models/yolo.py文件，在models文件内的yolo.py内将Detect类中的forward函数注释并且替换成如下图该函数，注意我们将该forward函数替换后下一次训练模型的时候一定要替换回去，不然会导致训练模型时报错，也会导致detect.py运行不了（detect.py是yolov5的检测代码，用来调用pt权重文件来去检测）




 

将export.py内的run函数内的shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  改为 shape = tuple(y[0].shape)  
然后在export.py中找到parse_opt函数中看下图：
 
翻到最后parse_opt函数处：--data参数指定的.yaml文件（如coco128.yaml ）中通常包含数据集的类别信息，比如类别名称和类别数量。在模型转换过程中，特别是转换为一些特定格式（如 Tensor - RT、ONNX 等）后用于实际推理时，模型需要知道检测目标的类别信息。例如，在推理结果的后处理阶段，需要根据这些类别信息来正确标注检测到的物体类别。
有些.yaml数据配置文件中还可能包含锚框（anchor boxes）相关的配置信息。虽然锚框主要在训练阶段使用较多，但在一些模型转换和推理的场景中，也可能会涉及到锚框相关的处理，比如在某些自定义的后处理逻辑中，需要依据数据集的锚框配置来对检测结果进行调整等。
注意：！！！综上也就是说在模型转换时我们仍然需要指定我们的.pt模型对应的数据集文件
例如：yolov5s.pt对应他们官方的coco128.yaml数据集
我们自己的 .pt模型对应我们自己的myvoc.yaml数据集

所以我们如果是自己的模型转化，则需要更改第三行default=ROOT / 后单引号内的路径，改成自己的数据集文件路径。然后将第二行内的.pt模型名改为自己在当前目录下要转化为onnx的模型文件名。

终端输入 python export.py --include onnx  即可将pt转为onnx。


onnx转rknn：
在上面的步骤中我们将pt转为onnx，接下来我们就需要执行我们在电脑端的最后一步，将onnx转为开发板可部署的rknn模型

由于onnx模型无法匹配瑞芯微rk芯片的npu，所以我们需要利用瑞芯微官方工具rknntoolkit2工具包  来去进行模型转化，首先我们需要获取到该工具包，在本文档同目录下存放有各个版本的rknntoolkit2工具包，也可以去瑞芯微github官网下载，这里我给出链接：https://github.com/airockchip/rknn-toolkit2 进入该网站去下载你需要的版本的rknn工具包 
打开上述点击master，在点击Tag如下图，即可找到RKNN各版本 
我们下载新版也可以，下载老版也可以，前提是下载后要更新npu版本，rknntoolkit2工具包 内有npu的驱动，我们只需要将其内的驱动文件复制到开发板usr/lib文件内即可，具体后续板端部署会讲到，这里先提个醒

下载好rknntoolkit2后我们将其存放在你的子系统Ubuntu自己想放的位置，然后我们利用conda新建一个关于rknn的虚拟环境，具体指令    
conda create -n rknn python=3.x          
具体python=3.x 后的版本x根据自己的ubuntu安装的python版本去填写。
重要！（这里为什么要新建rknn虚拟环境，因为我们在使用yolov5或者rknn时会安装各类包或者软件，为了防止这些包直接安装在我们的ubuntu系统内导致冲突或者安装失败导致系统环境紊乱，所以我们需要给yolov5创建属于他的虚拟环境，给rknn创建属于他的虚拟环境，两者各自安装的包都不会与对方冲突，也就是说，我们可以在yolov5虚拟环境下安装opencv，也可以在rknn虚拟环境下安装opencv，当然不进入这两个虚拟环境的话，系统内也是没有opencv的，这样两者都不会对系统造成干扰，彼此互相也不干扰，保持系统纯净，这就是conda虚拟环境的目的）
使用rknntoolkit2
rknn虚拟环境新建好后我们进入其中，输入
conda activate rknn
然后再打开我们存放rknntoolkit2的文件夹 
rknn-toolkit2：是在电脑端部署的npu工具包，用来转换onnx模型，并且作为虚拟npu去在电脑上测试运行模型。
rknn-toolkit-lite2：是在开发板部署的npu工具包，用来在开发板上去运行推理模型。
rknpu2：里面存放有开发板npu的最新驱动，用来更新npu驱动

由于我们需要先转换模型，所以我们在rknn虚拟环境下打开 rknn-toolkit2文件夹，进入其中如下图






package：内存放的是使用rknntoolkit2所需要的环境依赖包
example：内存放的是rknn源代码（包括模型转换代码），我们需要运行里面的模型转换代码才可以将onnx转为rknn。
那么首先，如果我们想运行example内的模型转换代码，我们就先需要进入packages文件下安装我们运行rknntoolkit2模型转换代码所依赖的库。
打开packages，由于我们是在电脑安装所依赖库，所以进入x86_64文件夹内如下图 
根据自己的python版本去选择安装对应的版本
rknn_toolkit2-2.3.0-cp3x-cp3x-manylinux2014_x86_64：该文件用来安装rknntoolkit2环境。
requirements_cp3x-2.3.0.txt：用来安装rknntoolkit2环境所需要的一些依赖库


这时我们需要操作上述两个文件
首先安装rknntoolkit2环境所需要的依赖库
打开requirements_cp3x-2.3.0.txt文档，里面第一行告诉我们了，我们需要使用清华源来对这个文档内的依赖库进行安装
终端输入：pip install -r requirements_cp3x-2.3.0.txt  即可安装该文档内所有写出来的依赖包，如果下载缓慢，就要按照人家第一行说的，使用清华源下载
终端输入：pip install -r requirements_cp3x-2.3.0.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/                           
即可安装rknntoolkit2环境所需依赖库成功

其次安装rknntoolkit2环境

终端输入：pip install rknn_toolkit2-2.3.0-cp3x-cp3x-manylinux2014_x86_64.whl
即可安装rknntoolkit2环境成功
当上述两步都安装完毕后
终端进入python输入：from rknn.api import RKNN 如果不出现报错，说明我们的rknn环境及其所需依赖已经安装完毕，接下来我们可以转换模型了

将我们上一步转化得到的onnx文件放入rknn-toolkit2/examples/onnx/yolov5中
打开rknn-toolkit2/examples/onnx/yolov5/test.py文件，如下图 
红色序号1：我们刚放进去的onnx文件名称
红色序号2：转化后得到的rknn的文件名称
红色序号3：转化后需要调用验证图片，验证转化后模型检测效果的验证图片的路径
红色序号4：注意！！！！，在这一步我们需要打开当前目录下dataset.txt文件，将我们对应转化模型的验证图片的全称改写进去，不然转换失败。
总结上述四个序号：你转化哪个模型，就在红色序号3处填写该模型对应的验证图片的路径，顺道再将该验证图片的全称（xx.jpg）覆盖掉dataset.txt原来的内容。



再将下图所示CLASS中的类别名改成你自己的模型的类别名。
 








然后我们需要在test.py代码内添加sigmod函数，照着图片所示添加
 
然后再将怕process后处理函数用sigmod改写，具体怎么改写可以看上述图片红色序号2指向的函数内的注释，注释是你们默认的process函数内容，每段注释的上一行是我改写后的，也就是4个红色圈圈出来的部分。照上面的图片改写process后处理函数即可。

然后我们在test.py中找到main函数中的rknn.config函数内的参数target_platform=“”， 双引号内填写你的RK系列开发板主控芯片型号，例如RK3588或者RK3566。

上述配置结束后再rknn虚拟环境下，在test.py目录下
终端输入python test.py 将我们的onnx模型转化为rknn模型
板端部署（开发板整个过程需联网）
1.	开发板Miniconda安装
Index of /anaconda/miniconda/ | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror   
打开上述链接进入清华源miniconda下载页面
根据自己开发板的python版本选择下载且由于miniconda安装在rk开发板，所以我们要找到arrch64后缀的miniconda包去下载。
下载好后将miniconda传入开发板，我们在开发板系统桌面新建一个文件夹，命名为conda，将下载得到的miniconda包放进去（如果你不想新建conda都可以，直接将Miniconda放在桌面，安装也行），然后在终端打开conda文件夹，使用
sudo chmod +x Miniconda的包文件名->如下图
 
赋予脚本执行权限，然后输入你的系统用户密码。

接下来使用sudo /.Miniconda的包文件名  来去安装Miniconda，如下图 
在安装miniconda的时候，有时候她会默认安装到root用户下，导致我们找不到，如下图1号箭头所示，所以我们需要在2号所示箭头处填写我们要让miniconda安装的位置。
   


安装完成后输入conda -V来去检测conda有无正常安装，如果报错或者检测不到，则我们需要手动添加环境变量，打开终端，输入nano ~/.bashrc 进入后到最后一行，添加如下图所示指令，注意，具体路径需要根据你自己的Miniconda安装路径确定，不能直接照搬我下图。 
注意在 export PATH=”x/x/x/x/bin:PATH” 除过中间标红的内容都是由你自己的miniconda安装路径确定的，其他一致，添加完成后保存退出，输入 
source ~/.bashrc 后重启终端，再次输入conda-V，如果出现你的虚拟环境数，说明conda目前已经正常安装。


然后使用conda create -n 你的虚拟环境名 python=3.7 记住，这里的python=3.7，3.7指的是你想要让你的虚拟环境中的python版本是多少，而不是你的开发板系统中的python版本，当然你可以指定虚拟环境中的python版本和开发板中的保持一致，只需要知道开发板系统的python版本，那么在这里我们就可以将python=3.x换成开发板系统的python版本，保持一致性，不然后面遇到需要根据python版本去安装对应的包的时候容易弄混！！！！！！！！当然如果你需要在你创建的虚拟环境中安装指定python版本的依赖包，那么还是需要根据虚拟环境当中的python版本来去安装的，比如我们后续会在创建好的虚拟环境中安装rknntoolkit-lite2，需要选择对应python版本的rknntoolkit-lite2，在这个时候考虑的python版本就是虚拟环境当中的python版本。

创建虚拟环境时可能会很慢，建议将开发板下载源变为清华源去创建虚拟环境。

所以我们现在创建一个你自己的虚拟环境，在该虚拟环境中去安装rknntoolkit-lite2工具包
如下图 
然后进入创建好的rknn虚拟环境中 conda activate rknn 
！！！！！！！！！！！！！！！
将我们在模型转化章节下载的rknntoolkit2工具包中的rknntoolkit-lite2和rknpu2这两个文件夹全部传到开发板桌面上
在rknn虚拟环境中，我们打开开发板桌面上的rknntoolkit-lite2/packages/文件夹
 
在上图packages文件内的rknn工具包cp后的版本号，找到与的与虚拟环境的python版本一致的rknn工具包

然后输入 pip install rknn_roolkit_lite2-x.x.x-cp3x-cp3x-linux_aarch64.whl 安装该工具包，x的内容根据你自己选择的rknn版本来确定。如果安装报错，极有可能是你选择的rknn工具包的版本与你的虚拟环境的版本不一致，或网络问题。

安装好后按照如下图所示验证rknnlite是否成功安装在开发板
 
无报错，说明安装成功。

接下来我们需要升级npu驱动，如果不升级，则会报错如下
 
或者会出现无法打开npu，需要手动加载npu的报错！！！！！
这是由于RKNN Runtime版本与RKNN-Toolkit-Lite2版本不一致导致的兼容性问题，所以我们需要升级RKNN Runtime版本，也就是更新npu驱动
在rknpu2文件rknpu2\runtime\Linux\librknn_api\aarch64目录找到librknnrt.so文件，将其复制放入开发板/lib/目录下，即可（在终端使用指令将其放进去，普通拖拽放不进去）
在rknpu2文件 rknpu2\runtime\Linux\rknn_server\aarch64\usr\bin目录找到下图 
红色箭头所示三个文件，这三个是打开npu服务和链接npu的驱动文件，我们在该目录下打开终端，将这三个文件复制放入/usr/bin目录下
然后我们进入开发板usr/bin目录下，在该目录下打开终端
依次输入：sudo chmod +x restart_rknn.sh
          sudo chmod +x rknn_server
          sudo chmod +x start_rknn.sh 分别赋予上述三个文件执行权限
紧接着然后我们在终端直接输入 restart_rknn.sh 如下图，当出现如下图第二行信息
 
反馈时，说明我们的npu能够正常运行且驱动已经安装完毕了，接下来我们才可以开始下一步。

然后再安装opencv和numpy
终端输入：sudo install opencv-python  来去安装opencv，至于numpy可以先查看下numpy的版本，看下是否有numpy，有的话就不用安装了，没有的话输入
sudo install numpy  来去安装。
上述完成后，我在本文档目录下有写好的板端推理代码，在 /已经完善好的项目/ 目录下找到，我们即可将我们训练好的rknn模型与开发板端推理代码放入同一文件夹放到开发板上。
 
在rknn虚拟环境下（因为虚拟环境安装了rknn工具包，可以使用npu加速推理，哈哈其实都没必要说，肯定懂），打开推理代码 
IMG_PATH是验证图片的存放位置，用来验证模型是否能正确推理检测该验证图片，运行后推理代码后得到result.jpg图片表征结果。
将红圈内的模型文件改成自己转化好的模型文件，然后找到
ret = rknn.init_runtime() 代码段，将其修改为
ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)使用开发板的三个npu共同推理，这样帧率会提高些。

注意：我们在开发板端的推理代码中，post-process（后处理）函数中的锚点anchors的具体参数数值应当与电脑端训练时yolov5s.yaml中的锚点参数保持一致，如果不一致，会造成检测框忽大忽小

在终端使用python运行该推理代码，即可实现模型推理检测，如下图 
 
后续我们可以在推理代码基础上增加多线程及提高npu和cpu利用率来增加帧率，因为目前3个npu根本没跑满，两个占用为0%，一个占用为10%，无法发挥出RK3588该有的性能。

