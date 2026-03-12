function voxel = readVoxelModel(filename)
    % 从 HDF5 文件中读取 voxel 模型
    % 获取文件中的所有键
    info = h5info(filename);
    if isempty(info)
        error('HDF5 文件为空或不存在');
    end
    
    % 获取所有组的名称
    groups = info.GroupHierarchy.Groups;
    if isempty(groups)
        error('HDF5 文件中没有找到任何组');
    end
    
    % 选择第一个组下的 voxel 数据集
    firstKey = groups(1).Name;
    voxelPath = [firstKey '/voxel'];
    
    % 检查路径是否存在
    if ~isfield(groups(1), 'Datasets') || ~any(strcmp({groups(1).Datasets.Name}, 'voxel'))
        error('HDF5 文件中没有找到 voxel 数据集');
    end
    
    % 从 HDF5 文件中读取 voxel 模型
    voxel = h5read(filename, voxelPath);
    % 确保 voxel 是一个逻辑矩阵（0 或 1）
    voxel = logical(voxel);
end