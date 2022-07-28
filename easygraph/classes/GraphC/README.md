## Boost Python 使用

### Windows

#### 安装

部分可参考[B站视频](https://www.bilibili.com/video/BV1YE411w722)，或[简书](https://www.jianshu.com/p/2cabb894404e)

1. 确保VS已安装MSVC编译器和Win10 SDK

   ![image-20220711004442554](https://s2.loli.net/2022/07/11/Y1I3Pg6fcwpRhx5.png)

2. 在VS的Develop Command Prompt终端中**进入[下载](https://www.boost.org/users/history/version_1_79_0.html)好的Boost文件夹**

   <img src="https://s2.loli.net/2022/07/16/bzhkCV2SrqXEP5d.png" alt="image-20220716113520507" style="zoom: 67%;" />

   **注意**：

   - 编译出的lib库与当前环境下的python是匹配的，所以如果要为anaconda下某个python编译，需要在Develop Command Prompt中conda activate ...

3. 运行bootstrap.bat编译出b2.exe

    ```
    bootstrap.bat
    
3. **根据本机环境**(例如有conda等)修改如下命令并运行：

    ```
    b2 install --toolset=msvc-14.3 --with-python --prefix="D:\Program Files (x86)\Boost" link=static runtime-link=shared threading=multi address-model=64
    ```

    **注意**：
    
    - 这里toolset的msvc版本需要对应自己安装VS的msvc版本，我这里是14.3
    - prefix是编译后文件的存储路径
    - link选择static
    - address-model选择与python版本对应的，例如64为python就选择64

#### 使用

##### 依赖配置

**注意**：以下步骤需要将解决方案生成模式切换为**Release模式**，Debug模式下IDE会添加大量冗余代码，造成性能大幅下降

![image-20220716113937603](https://s2.loli.net/2022/07/16/Fszj27ecQ5iAMJl.png)

项目属性

- 常规
  - 配置类型：动态库(.dll)
  - ![image-20220711004156641](https://s2.loli.net/2022/07/11/wUQfeBIpAjlJgMC.png)
- 高级
  - 目标文件拓展名：.pyd
  - ![image-20220711004259682](https://s2.loli.net/2022/07/11/QnCsOJTBM9Wbo5m.png)
- VC++目录
  - 包含目录：目标环境下python的`include`文件夹、上述b2编译生成路径中的`include\boost-xxx`文件夹
  - 库目录：目标环境下python的`libs`文件夹、上述b2编译生成路径中的`lib`文件夹
  - ![image-20220711004047611](https://s2.loli.net/2022/07/11/7CxRNuLj3To1fV9.png)

##### 代码编写

boost::python::extract\<T>(...)  负责从参数中提取出类型T，这是常用的类型转换方式

其余请参考Wiki、各问答网站、及博客，另外也可参考Bilibili上视频教程

**注意**：

- **每个头文件**中都应写上`#define BOOST_PYTHON_STATIC_LIB`，否则链接时会出现找不到python_xxx.lib【b2编译阶段选择link=shared才会生成】的报错

##### 生成使用

直接点击生成解决方案即可。确认生成目录下(`根目录/x64/Release/`)成功生成.pyd文件后尝试import。


### Mac

#### 安装

因为官方网上的教程过于简单，本人多次尝试均出现各种奇怪的问题，所以建议直接使用mac平台下的软件包工具:homebrew。

该教程默认电脑上是有python3的。

Homebrew下载：

官方网址：https://brew.sh/

进入官方网后，直接复制Install Homebrew标题下的命令，然后打开mac的终端执行即可。

用Homebrew下载软件包是在命令行执行的，一般格式是：brew install xxx

更新Homebrew的命令：brew update

其他homebrew特性可以在官方网或其他论坛上查看。

用homebrew下载boost,下面是两条命令，请依次执行，第一条命令第一次执行比较慢，请耐心等待。

```
brew install boost --build-from-source
brew install boost-python3
```

homebrew一般会把下载的这些软件包放在:/usr/local/Cellar/目录下，可以command+shift+g查看是否存在boost，boost-python3

下载完成后，首先新建一个hello_ext.cpp文件测试。

```C++
#include <boost/python.hpp>

char const* greet()
{
   return "Greetings!";
}

BOOST_PYTHON_MODULE(hello_ext)
{
    using namespace boost::python;
    def("greet", greet);
}
```

然后在当前目录下新建一个setup.py文件用于构建拓展

```python
from distutils.core import setup
from distutils.extension import Extension

hello_ext = Extension(
    'hello_ext',
    sources=['hello_ext.cpp'],
    libraries=['boost_python-mt'],
)

setup(
    name='hello-world',
    version='0.1',
    ext_modules=[hello_ext])
```

然后在命令行运行

```shell
python setup.py build_ext --inplace
```

然后会看到当前目录下出现

```
build/
hello_ext.xxxx.so
```

其中.so文件就是生成的调用库，可以进入python命令行直接

```
In [1]: import hello_ext
In [2]: hello_ext.greet()
Out[2]: 'Greetings!‘
```

若成功出现'Greetings!‘则说明boost_python3构建成功。

若出现:ld: library not found for -lboost_pythonxx(xx代表你电脑对应默认python版本，比如我的电脑是python3.9,则xx就是39)的问题，则首先在你的电脑中查找是否有libboost_python39.dylib这个文件。

若没有：则说明你brew install boost-python3没成功，可以再下载,注意下载过程中是否有报错，若报错了则对应修改或执行相应推荐命令即可。下载完成后可以通过再执行brew install boost-python3，若出现告诉你boost-python3没link，则按照提示去link即可。

若有：则试试 brew link boost-python3,若出现权限不够的问题，可以利用chmod修改对应文件的权限为777后再执行，或者你不希望所有用户有读写这些文件的权利，可以查询其他数字组合

```shell
chmod 777 文件绝对路径
```

在brew link boost-python3后若出现Warning: Already linked: /usr/local/Cellar/boost-python3/1.79.0，则代表已经link过了。

若成功应该可以在/usr/local/lib/下看到libboost_python39.dylib，且图标下方有个小箭头，代表软链接到了/usr/local/Cellar/boost-python3/1.79.0/lib中的libboost_python39.dylib

![截屏2022-07-16 09.13.03.png](https://s2.loli.net/2022/07/16/yg3we6KisonH7CG.png)

若出现其他不可预见的问题，可以在csdn/stackoverflow上查找答案。

Linux

#### 安装

部分可参考[稀土掘金]([Linux安装Boost Python - 掘金 (juejin.cn)](https://juejin.cn/post/6870325642362093582))

1. 确保系统已安装
  
  - **gcc**
    
  - **g++**
    
  - **python3-dev**
    
  
  以ubuntu为例，可以运行如下命令安装：
  
  ```shell
     sudo apt-get install ...
  ```
  
  **注意**：
  
  - python3-dev用于获取必要的python头文件和静态库，对于较新版本的python(3.9+)，可能需要从[**其他发布平台**](https://pkgs.org/download/python3.9-dev)获取安装对应版本python3-dev
2. 下载boost源码并解压
  
3. 进入boost目录，执行boostrap.sh，生成b2
  
  ```shell
  ./bootstrap.sh --with-python=/usr/bin/python3
  ```
  
  编译完成后将生成b2可执行文件
  
4. 编译安装boost python
  
  ```shell
  sudo ./b2  cxxflags="-fPIC" install --with-python
  ```
  
  **注意**：
  
  - 安装过程涉及对高权限目录的更改，因此需要sudo
    
  - cxxflags用于编译boost python静态库
    
5. 进入/usr/local/lib确认生成libboost_pythonxx.a, libboost_pythonxx.so
  
  将libboost_pythonxx.a软链接到libboost_python.a
  
  ```shell
  sudo ln -s libboost_python38.a libboost_python.a
  ```
  
  **注意**：
  
  - 软链接的目的是为了给静态库换名，gcc在链接时会优先选择同名的动态库，换名后**没有**libboost_python.so，因此链接boost_python时只会选择已有的libboost_python.a
    
6. 编译生成项目文件
  
  ```shell
  rm *.o
  rm *.so
  g++ -fPIC -shared -I/usr/include/python3.8 -Wl,-soname,cpp_easygraph.so -o cpp_easygraph.so  Graph.cpp Utils.cpp Evaluation.cpp Path.cpp cpp_easygraph.cpp -lpython3.8 -lboost_python
  # 可以添加-o..等编译优化标志
  ```
  
  **注意**：
  
  - -I后面是python的头文件目录，已知有时gcc无法从环境变量中搜索到相关文件，因此这里手动指定
    
  - -fPIC表明生成的是位置无关的动态链接库
    
  - -lpython3.8 和 -lboost_python的**顺序**不能颠倒(gcc从右向左链接)，否则有符号问题
    
  - 如上文所述，这里-lboost_python实际找到的是libboost_python.a(静态boost python库)，若链接到的是.so，则导入时会提示找不到相关so文件
    

#### 使用

在linux中使用C++全局py::object变量(无论多文件共享或单个文件独有)导入模块时会有segmentation fault的风险，因此不建议使用。
