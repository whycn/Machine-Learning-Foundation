function two_d_grad_wrapper_hw()
run_all()
%%%%%%%%%%%% subfunctions %%%%%%%%%%%%
%%% performs Newton steps %%%%
    function [w,in,out] = gradient_descent(alpha,w)
    % initializations
    grad_stop = 10^-5;
    max_its = 10;
    iter = 1;
    grad = 1;
    in = [w];
     out = [log(1+exp(w'*w))];
    % main loop
    while norm(grad) > grad_stop && iter <= max_its
        % take gradient step
        grad =w*(1+exp((w'*w)))/(1+exp((w'*w))+2*w'*w);   %%% PLACE GRADIENT HERE
        w = w - grad;
 
        % update containers
        in = [in, w];
        out = [out, log(1+exp(w'*w))];
 
        % update stopers
        iter = iter + 1;
    end
end
function run_all()
 
       x0 = [1;1];         % initial point (for gradient descent) 
            alpha = 2*10^-3;
        %%% perform gradient descent %%%
        [x,in,out] = gradient_descent(alpha,x0);
 
        %%% plot function with grad descent objective evaluations %%%
        hold on
        plot_it_all(in,out)
end
 
%%% plots everything %%%
function plot_it_all(in,out)
    % print function
    [A,b] = make_fun();
    
    % print steps on surface
    plot_steps(in,out,3)
    set(gcf,'color','w');
end
 
%%% plots everything %%%
function [A,b] = make_fun()
    range = 4.15;                     % range over which to view surfaces
    [a1,a2] = meshgrid(-range:0.04:range);
    a1 = reshape(a1,numel(a1),1);
    a2 = reshape(a2,numel(a2),1);
    A = [a1, a2];
    A = (A.*A)*ones(2,1);
    b = log(1+exp(A))
    r = sqrt(size(b,1));
    a1 = reshape(a1,r,r);
    a2 = reshape(a2,r,r);
    b = reshape(b,r,r);
    h = surf(a1,a2,b)
    az = 35;
    el = 60;
    view(az, el);
    shading interp
 
    
    xlabel('w_1','Fontsize',18,'FontName','cmmi9')
    ylabel('w_2','Fontsize',18,'FontName','cmmi9')
    zlabel('g','Fontsize',18,'FontName','cmmi9')
    set(get(gca,'ZLabel'),'Rotation',0)
    set(gca,'FontSize',12);
    box on
    colormap gray
end
 
% plot descent steps on function surface
function plot_steps(in,out,dim)
    s = (1/length(out):1/length(out):1)';
    colorspec = [ones(length(out),1) ,zeros(length(out),1),flipud(s)];
    width = (1 + s)*5;
    if dim == 2
        for i = 1:length(out)
            hold on
            plot(in(1,i),in(2,i),'o','Color',colorspec(i,:),'MarkerFaceColor',colorspec(i,:),'MarkerSize',width(i));
        end
    else % dim == 3
        for i = 1:length(out)
            hold on
            plot3(in(1,i),in(2,i),out(i),'o','Color',colorspec(i,:),'MarkerFaceColor',colorspec(i,:),'MarkerSize',width(i));
        end
    end
end
end
