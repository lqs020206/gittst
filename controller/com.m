clear all;

%% 打开端口
if ~isempty(instrfind('Port', 'COM9'))      % 检查是否有其他设备已经连接到该端口
    fclose(instrfind('Port', 'COM9'));      % 关闭其他设备的连接
    delete(instrfind('Port', 'COM9'));      % 删除其他设备的连接对象
end
s = serial('COM9', 'BaudRate', 9600);       % 创建一个串口对象
try
    fopen(s);  % 打开 COM9 端口
catch
    error('Failed to open serial port.');   % 如果打开失败，抛出错误
end

%% 发送数据
for i = 1:10
    data = zeros(1,46);     % 生成指令
    data(1,46) = 128;       % 0x80 结束
    for id = 10 % 0-128 正，256-128 负，越靠近 128 速度越大
        data(1,3 * id - 2) = 0;      % vy 这里 y 和 x 是反过来的
        data(1,3 * id - 1) = 20;     % vx
        data(1,3 * id) = 0;          % omega
    end
    fwrite(s, data);        % 发送数据
    pause(0.5);             % 暂停 0.5s
end
% 结束时发零
data = zeros(1,46);
data(1,46) = 128;
fwrite(s, data);

%% 关闭串口相关操作
fclose(s);  % 关闭串口
delete(s);  % 删除内存中的串口对象
