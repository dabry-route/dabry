function main()
    fname = 'input/args.json';
    fid = fopen(fname); 
    raw = fread(fid,inf); 
    str = char(raw'); 
    fclose(fid); 
    val = jsondecode(str);
    t_start = val(1).t_start;
    t_end = val(1).t_end;
    x_min = val(1).x_min;
    y_min = val(1).y_min;
    x_max = val(1).x_max;
    y_max = val(1).y_max;
    
    x_init = val(1).x_init;
    y_init = val(1).y_init;

    airspeed = val(1).airspeed;

    wind = load('input/wind.mat').data;
    sz = size(wind);
    nx = sz(2);
    ny = sz(3);
    nt = sz(4);
    wind_u = reshape(wind(1, :, :, :), nx, ny, nt);
    wind_v = reshape(wind(2, :, :, :), nx, ny, nt);
    %wind_u = table2array(readtable('~/Documents/work/level-set/input/u.txt', 'Delimiter', 'comma'));
    %wind_v = table2array(readtable('~/Documents/work/level-set/input/v.txt', 'Delimiter', 'comma'));
    
    tol = 2 * min((x_max - x_min) / (nx), (y_max - y_min) / (ny));
    
    display=true;
    if ~isprop(val(1), 'display')
        display=false;
    end
    
    run('~/Documents/work/ToolboxLS/Examples/addPathToKernel.m');
    
    %---------------------------------------------------------------------------
    % Speed of motion normal to the interface.
    aValue = airspeed;
    
    %---------------------------------------------------------------------------
    % Integration parameters.
    tMax = t_end;                  % End time.
    plotSteps = val(1).nt_rft;               % How many intermediate plots to produce?
    t0 = t_start;                      % Start time.
    singleStep = 0;              % Plot at each timestep (overrides tPlot).
    
    % Period at which intermediate plots should be produced.
    tPlot = (tMax - t0) / (plotSteps - 1);
    
    % How close (relative) do we need to get to tMax to be considered finished?
    small = 100 * eps;
    
    %---------------------------------------------------------------------------
    % What level set should we view?
    level = 0;
    
    % Pause after each plot?
    pauseAfterPlot = 0;
    
    % Delete previous plot before showing next?
    deleteLastPlot = 0;
    
    % Plot in separate subplots (set deleteLastPlot = 0 in this case)?
    useSubplots = 1;
    
    %---------------------------------------------------------------------------
    % Use periodic boundary conditions?
    periodic = 0;
    
    % Create the grid.
    g.dim = 2;
    g.min = [x_min; y_min];
    g.max = [x_max; y_max];
    g.N = [nx; ny];
    %g.dx = (x_max - x_min) / (nx - 1);
%     if(periodic)
%       g.max = (1 - g.dx);
%       g.bdry = @addGhostPeriodic;
%     else
%       g.max = x_max;
%       g.bdry = @addGhostExtrapolate;
%     end
    g.bdry = @addGhostExtrapolate;
    g = processGrid(g);
    
    %---------------------------------------------------------------------------
    % What kind of display?
    if(nargin < 3)
      switch(g.dim)
       case 1
        displayType = 'plot';
       case 2
        displayType = 'contour';    
       case 3
        displayType = 'surface';
       otherwise
        error('Default display type undefined for dimension %d', g.dim);
      end
    end
    
    %---------------------------------------------------------------------------
    % Create initial conditions (star shaped interface centered at origin).
    %   Note that in the periodic BC case, these initial conditions will not be
    %   continuous across the boundary.  Regardless of boundary conditions, this
    %   initial function will be far from signed distance (although it is
    %   definitely an implicit surface function).  In practice, we'll just
    %   ignore these little details.
    data = sqrt((g.xs{1} - x_init * ones(g.shape)).^2 + (g.xs{2} - y_init * ones(g.shape)).^2) - tol;
    data0 = data;
    
    %---------------------------------------------------------------------------
    if(nargin < 1)
      accuracy = 'medium';
    end
    
    % Set up time approximation scheme.
    integratorOptions = odeCFLset('factorCFL', 0.5, 'stats', 'on');
    
    % Choose approximations at appropriate level of accuracy.
    %   Same accuracy is used by both components of motion.
    switch(accuracy)
     case 'low'
      derivFunc = @upwindFirstFirst;
      integratorFunc = @odeCFL1;
     case 'medium'
      derivFunc = @upwindFirstENO2;
      integratorFunc = @odeCFL2;
     case 'high'
      derivFunc = @upwindFirstENO3;
      integratorFunc = @odeCFL3;
     case 'veryHigh'
      derivFunc = @upwindFirstWENO5;
      integratorFunc = @odeCFL3;
     otherwise
      error('Unknown accuracy level %s', accuracy);
    end
    
    if(singleStep)
      integratorOptions = odeCFLset(integratorOptions, 'singleStep', 'on');
    end
    
    %---------------------------------------------------------------------------
    % Set up motion in the normal direction.
    normalFunc = @termNormal;
    normalData.grid = g;
    normalData.speed = aValue;
    normalData.derivFunc = derivFunc;
    
    function velocity = velocityFunc(t, ~, ~)
        index = 1 + round((nt-1) * (t - t_start) / (t_end - t_start));
        if(index < 0)
            index = 0;
        end
        if(index > nt - 1)
            index = nt - 1;
        end
        u = reshape(wind(1, :, :, index), nx, ny);
        v = reshape(wind(2, :, :, index), nx, ny);
        velocity = {u; v};
    end
    rotationFunc = @termConvection;
    rotationData.grid = g;
    rotationData.derivFunc = derivFunc;
    
    rotationData.velocity = @velocityFunc; %{wind_u; wind_v};
    
    %---------------------------------------------------------------------------
    % Combine components of motion.
    schemeFunc = @termSum;
    schemeData.innerFunc = { normalFunc; rotationFunc };
    schemeData.innerData = { normalData; rotationData };
    
    %---------------------------------------------------------------------------
    % Initialize Display
    if(display)
        f = figure;
    
        % Set up subplot parameters if necessary.
        if(useSubplots)
          rows = ceil(sqrt(plotSteps));
          cols = ceil(plotSteps / rows);
          plotNum = 1;
          subplot(rows, cols, plotNum);
        end
        
        h = visualizeLevelSet(g, data, displayType, level, [ 't = ' num2str(t0) ]);
        
        hold on;
        if(g.dim > 1)
          axis(g.axis);
          daspect([ 1 1 1 ]);
        end
    end
    
    %---------------------------------------------------------------------------
    % Loop until tMax (subject to a little roundoff).
    tNow = t0;
    startTime = cputime;
    i = 0;
    writematrix(data, ['output/rff_' num2str(i) '.txt'])
    i = i + 1;
    timesteps = [];
    timesteps(end + 1) = t_start;
    while(tMax - tNow > small * tMax)
    
      % Reshape data array into column vector for ode solver call.
      y0 = data(:);
    
      % How far to step?
      tSpan = [ tNow, min(tMax, tNow + tPlot) ];
      
      
      % Take a timestep.
      [ t y ] = feval(integratorFunc, schemeFunc, tSpan, y0,...
                      integratorOptions, schemeData);
      tNow = t(end);
      timesteps(end + 1) = tNow;
    
      % Get back the correctly shaped data array
      data = reshape(y, g.shape);
      
      if(display)
          if(pauseAfterPlot)
            % Wait for last plot to be digested.
            pause;
          end
        
          % Get correct figure, and remember its current view.
          figure(f);
        
          % Delete last visualization if necessary.
          if(deleteLastPlot)
            delete(h);
          end
        
          % Move to next subplot if necessary.
          if(useSubplots)
            plotNum = plotNum + 1;
            subplot(rows, cols, plotNum);
          end
        
          % Create new visualization.
          h = visualizeLevelSet(g, data, displayType, level, [ 't = ' num2str(tNow) ]);
      end
      % Write level set function to file
      writematrix(data, ['output/rff_' num2str(i) '.txt'])
      i = i + 1;
      
    end
    writematrix(timesteps, 'output/timesteps.txt')
    
    endTime = cputime;
    fprintf('\nTotal execution time %g seconds\n', endTime - startTime);
end
