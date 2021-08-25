close all;
clear all;

t_server = tcpip('192.168.1.116',5555,'NetworkRole','server');
t_server.InputBuffersize = 1000000;
fopen(t_server);                                          %打开服务器，直到建立一个TCP连接才返回；
get(t_server);
fwrite(t_server,'123')
while(1)
    if  t_server.BytesAvailable > 0
        t_server.BytesAvailable;
        data = strcat(fread(t_server,t_server.BytesAvailable));
        d = eval(data);
        data = '';
        g = d{1};
        x = d{2};
        y = d{3};
        z = d{4};
        s = d{5};
        g
        clf;
        plot3([0,g(1)],[0,g(2)],[0,-g(3)]);
        hold on;
        plot3(x,y,z);
        plot3(s(1),s(2),s(3),'o');
        xlim([-20,20]);
        ylim([-20,20]);
        zlim([-20,20]);
        grid on
        pause(1/25);
        fwrite(t_server,'123')
    end
end
