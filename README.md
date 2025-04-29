ZYB用于自搭建双目相机的工具箱

# 标定
采用matlab，先用双目标定工具箱，导出参数之后运行camera_calib.m就可以保存俩yaml

后续操作运行test_shuangmu.py

$$\begin{array}{l}
m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) \nabla J\left(\theta_{t}\right)\\
v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) \nabla J\left(\theta_{t}\right)^{2}\\
\hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}}, \quad \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}}\\
\theta_{t+1}=\theta_{t}-\frac{\alpha}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}
\end{array}$$