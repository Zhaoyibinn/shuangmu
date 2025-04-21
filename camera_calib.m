%%%%%% ===============   输出外参文件   =====================
output = fopen('extrinsics.yml', 'wt');
fprintf(output, '%%YAML:1.0\n');
fprintf(output, '---\n');
%R:双目外参-旋转矩阵
fprintf(output, 'R: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n   data: [ ');
R = stereoParams.RotationOfCamera2;
for jj=1:8
    fprintf(output, '%.14f, ', R(jj));
end
fprintf(output, '%.14f', R(9));
fprintf(output, ' ]\n');
%T:双目外参-平移向量
fprintf(output, 'T: !!opencv-matrix\n   rows: 3\n   cols: 1\n   dt: d\n   data: [');
T = stereoParams.TranslationOfCamera2;
for jj=1:length(T)-1
    fprintf(output, '%.14f, ', T(jj));
end
fprintf(output, '%.14f', T(length(T)));
fprintf(output, ' ]\n');
%Rw:手眼标定参数-旋转矩阵
fprintf(output, 'Rw: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n   data: [ ');
fprintf(output, '1, 0, 0, 0, 1, 0, 0, 0, 1');
fprintf(output, ' ]\n');
%Tw:手眼标定参数-平移向量
fprintf(output, 'Tw: !!opencv-matrix\n   rows: 3\n   cols: 1\n   dt: d\n   data: [ ');
fprintf(output, '0, 0, 0');
fprintf(output, ' ]\n');
%R1:未使用
fprintf(output, 'R1: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n   data: [ ');
fprintf(output, '1, 0, 0, 0, 1, 0, 0, 0, 1');
fprintf(output, ' ]\n');
%R2:未使用
fprintf(output, 'R2: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n   data: [ ');
fprintf(output, '1, 0, 0, 0, 1, 0, 0, 0, 1');
fprintf(output, ' ]\n');
%P1:未使用
fprintf(output, 'P1: !!opencv-matrix\n   rows: 3\n   cols: 4\n   dt: d\n   data: [ ');
fprintf(output, '1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1');
fprintf(output, ' ]\n');
%P2:未使用
fprintf(output, 'P2: !!opencv-matrix\n   rows: 3\n   cols: 4\n   dt: d\n   data: [ ');
fprintf(output, '1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1');
fprintf(output, ' ]\n');
%Q:未使用
fprintf(output, 'Q: !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: d\n   data: [ ');
fprintf(output, '1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1');
fprintf(output, ' ]\n');
%F:基础矩阵Fundamental Matrix
fprintf(output, 'F: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n   data: [ ');
F = stereoParams.FundamentalMatrix;
for jj=1:8
    fprintf(output, '%.14f, ', F(jj));
end
fprintf(output, '%.14f', F(9));
fprintf(output, ' ]\n');

fclose(output);




%%%%%% ===============   输出内参文件   =====================
output = fopen('intrinsics.yml', 'wt');
fprintf(output, '%%YAML:1.0\n');
fprintf(output, '---\n');
%M1：左相机的内参矩阵
fprintf(output, 'M1: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n   data: [ ');
M1 = stereoParams.CameraParameters1.IntrinsicMatrix;
for jj=1:8
    fprintf(output, '%.14f, ', M1(jj));
end
fprintf(output, '%.14f', M1(9));
fprintf(output, ' ]\n');
%D1: 左相机畸变系数
fprintf(output, 'D1: !!opencv-matrix\n   rows: 1\n   cols: 5\n   dt: d\n   data: [ ');
RaD1 = stereoParams.CameraParameters1.RadialDistortion;
TaD1 = stereoParams.CameraParameters1.TangentialDistortion;
fprintf(output, '%.14f, %.14f, %.14f, %.14f, %.14f ]\n', RaD1(1), RaD1(2), TaD1(1), TaD1(2), 0);
%M2：右相机的内参矩阵
fprintf(output, 'M2: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n   data: [ ');
M2 = stereoParams.CameraParameters2.IntrinsicMatrix;
for jj=1:8
    fprintf(output, '%.14f, ', M2(jj));
end
fprintf(output, '%.14f', M2(9));
fprintf(output, ' ]\n');
%D2: 右相机畸变系数
fprintf(output, 'D2: !!opencv-matrix\n   rows: 1\n   cols: 5\n   dt: d\n   data: [ ');
RaD2 = stereoParams.CameraParameters2.RadialDistortion;
TaD2 = stereoParams.CameraParameters2.TangentialDistortion;
fprintf(output, '%.14f, %.14f, %.14f, %.14f, %.14f ]\n', RaD2(1), RaD2(2), TaD2(1), TaD2(2), 0);

fclose(output);