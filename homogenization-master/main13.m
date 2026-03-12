%% 主程序：均质化本构计算
clear; clc; close all;

% ========== 用户自定义参数 ==========
h5File = 'voxel_models_batch_13.h5';   % HDF5文件路径
outputCSV = 'homogenized_results_13new1.csv'; % 输出CSV文件名
lambda = 115.4;                    % Lame参数lambda
mu = 76.9;                         % Lame参数mu
lx = 2.0; ly = 2.0; lz = 2.0;      % 单元格尺寸
useParallel = false;               % 是否启用并行计算（暂时禁用）
% ==================================

%% 初始化
% 添加homo3D函数路径（如果不在当前目录）
% addpath('/path/to/homo3D/directory');

% 启动并行池（如果启用并行计算）
if useParallel
    pool = gcp('nocreate');
    if isempty(pool)
        parpool('local'); % 根据CPU核心数自动启动并行池
    end
end

%% 读取HDF5文件中的体素模型
try
    info = h5info(h5File);
    keys = {info.Groups.Name}; % 获取所有模型键名（如 '/model_1', '/model_2' ...）
catch ME
    error('HDF5文件读取失败: %s', ME.message);
end

%% 调试信息：打印 keys 数组内容
disp('Keys in HDF5 file:');
disp(keys);

%% 并行处理所有体素模型
numModels = numel(keys);
fprintf('开始处理 %d 个体素模型...\n', numModels);

if useParallel
    resultsCell = cell(numModels, 1);
    parfor i = 1:numModels
        key = keys{i};
        fprintf('Processing model: %s\n', key);
        resultsCell{i} = processModel(i, key, h5File, lambda, mu, lx, ly, lz);
    end
    
    % 过滤掉无效的结果（确保结构体字段完整）
    validResultsCell = cellfun(@(x) isstruct(x) && numel(fieldnames(x)) == 7, resultsCell);
    resultsCell = resultsCell(validResultsCell);
    
    % 将 cell 数组转换为结构体数组
    if ~isempty(resultsCell)
        results = [resultsCell{:}];  % 直接合并结构体数组
    else
        results = struct(...
            'key', {}, ...
            'E1', [], 'E2', [], 'E3', [], ...
            'G12', [], 'G23', [], 'G13', []);
    end
    
    % 保存结果到CSV
    struct2csv(results, outputCSV);
    fprintf('结果已保存至: %s\n', outputCSV);
else
    % 初始化结果结构体（使用动态字段添加）
    results = struct('key', {}, 'E1', [], 'E2', [], 'E3', [], 'G12', [], 'G23', [], 'G13', []);
    
    for i = 1:numModels
        key = keys{i};
        fprintf('Processing model: %s\n', key);
        result = processModel(i, key, h5File, lambda, mu, lx, ly, lz);
        
        % 严格检查 result.key 是否为非空字符且 E1 有效
        if ischar(result.key) && ~isempty(result.key) && ~isnan(result.E1)
            try
                % 使用临时变量确保类型正确
                newKey = result.key;
                newE1 = result.E1;
                newE2 = result.E2;
                newE3 = result.E3;
                newG12 = result.G12;
                newG23 = result.G23;
                newG13 = result.G13;
                
                % 动态扩展结构体字段
                results(end+1).key = newKey;
                results(end).E1 = newE1;
                results(end).E2 = newE2;
                results(end).E3 = newE3;
                results(end).G12 = newG12;
                results(end).G23 = newG23;
                results(end).G13 = newG13;
                
                % 保存当前结果到CSV
                struct2csv(results, outputCSV);
                fprintf('中间结果已保存至: %s\n', outputCSV);
            catch ME
                fprintf('添加模型 %s 到结果时出错: %s\n', newKey, ME.message);
            end
        end
    end
end

%% 子函数：处理单个模型
function result = processModel(i, key, h5File, lambda, mu, lx, ly, lz)
    % 初始化结果结构体
    result = struct(...
        'key', '', ...
        'E1', NaN, 'E2', NaN, 'E3', NaN, ...
        'G12', NaN, 'G23', NaN, 'G13', NaN);
    
    % 确保 h5File 是字符串类型
    if ~ischar(h5File) && ~isstring(h5File)
        error('h5File 必须是字符或字符串类型');
    end
    
    % 确保 key 是字符串类型
    if ~ischar(key) && ~isstring(key)
        error('key 必须是字符或字符串类型');
    end
    
    % 调试信息：打印当前处理的 key 和 h5File
    fprintf('In processModel: key = %s, h5File = %s\n', key, h5File);
    
    % 检查 key 下的 /voxel 数据集是否存在
    try
        dsInfo = h5info(h5File, [key '/voxel']);
    catch ME
        fprintf('数据集 %s/voxel 不存在或无法访问: %s\n', key, ME.message);
        return;
    end
    
    % 读取体素数据
    try
        voxel = h5read(h5File, [key '/voxel']);
        
        % 检查是否为uint8类型且包含属性标记
        if isa(voxel, 'uint8') && isfield(dsInfo.Attributes, 'is_logical')
            voxel = logical(voxel); % 直接转换为logical
        elseif isnumeric(voxel)
            voxel = logical(voxel); % 强制转换数值类型
        elseif ~islogical(voxel)
            fprintf('voxel 数据不是数值类型或逻辑类型: %s\n', key);
            return;
        end
        
    catch ME
        fprintf('读取模型 %s 失败: %s\n', key, ME.message);
        return;
    end
    
    % 读取属性
    try
        density = h5readatt(h5File, key, 'density');
        active_faces = h5readatt(h5File, key, 'active_faces');
    catch ME
        fprintf('读取属性失败: %s\n', ME.message);
        return;
    end
    
    % 跳过空体素模型
    if sum(voxel(:)) == 0
        fprintf('跳过空模型: %s\n', key);
        return;
    end
    
    % 清除不需要的变量以释放内存
    clear dsInfo;
    
    % 转换为逻辑类型
    voxel = logical(voxel);
    
    % 预处理：检查 voxel 的连通性
    if ~checkConnectivity(voxel)
        fprintf('模型 %s 不连通，跳过计算。\n', key);
        return;
    end
    
    % 计算相对密度
    relative_density = sum(voxel(:)) / numel(voxel);
    
    % 调用均质化函数
    try
        result = homo3D(lx, ly, lz, lambda, mu, voxel);
        result.key = char(strrep(key, '/', ''));
        
        % 使用相对密度进行归一化
        result.E1 = result.E1 / relative_density;
        result.E2 = result.E2 / relative_density;
        result.E3 = result.E3 / relative_density;
        result.G12 = result.G12 / relative_density;
        result.G23 = result.G23 / relative_density;
        result.G13 = result.G13 / relative_density;
    catch ME
        fprintf('模型 %s 计算失败: %s\n', key, ME.message);
        result.key = char(strrep(key, '/', '')); % 确保result.key始终为字符数组
    end
    
    % 清除不再需要的变量
    clear voxel density active_faces;
    
    fprintf('已完成: %s (E1=%.4f)\n', key, result.E1);
end

%% 辅助函数：将结构体数组保存为CSV
function struct2csv(data, filename)
    tbl = struct2table(data);
    writetable(tbl, filename);
end

%% 辅助函数：检查 voxel 的连通性
function isConnected = checkConnectivity(voxel)
    % 使用 bwconncomp 检查 voxel 的连通性
    cc = bwconncomp(double(voxel));
    isConnected = (cc.NumObjects == 1);
end



