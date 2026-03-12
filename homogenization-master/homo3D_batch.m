function results = homo3D_batch(lx, ly, lz, lambda_, mu, voxel_matrices)
    % 函数功能：批量计算多个体素模型的均质化本构矩阵和各向异性模量
    % 输入参数：
    % lx: 单元格尺寸 (x方向)
    % ly: 单元格尺寸 (y方向)
    % lz: 单元格尺寸 (z方向)
    % lambda_: Lame参数
    % mu: Lame参数
    % voxel_matrices: 体素模型逻辑数组列表
    % 输出参数：
    % results: 包含所有模型计算结果的单元格数组，每个元素为一个结构体，
    %          结构体包含'CH'（均质化本构矩阵）、'E1'、'E2'、'E3'、'G12'、'G23'、'G13'（各向异性模量）
    
    num_models = length(voxel_matrices);
    results = cell(num_models, 1);
    for i = 1:num_models
        voxel = voxel_matrices{i};
        % 调用homo3D函数计算每个体素模型的结果
        result = homo3D(lx, ly, lz, lambda_, mu, voxel);
        results{i} = result;
    end
end