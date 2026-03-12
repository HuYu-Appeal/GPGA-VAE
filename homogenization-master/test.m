% 测试 homo3D 函数
lx = 1.0; ly = 1.0; lz = 1.0;
lambda = 115.4; mu = 76.9;
voxel = randi([0, 1], 100, 100, 100); % 示例体素模型
CH = homo3D(lx, ly, lz, lambda, mu, voxel);

% 检查返回值
disp(size(CH)); % 应该输出 [6, 6]
disp(CH);