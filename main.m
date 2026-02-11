% Context-Aware Image Captioning for Blind Assistance
% Integration of YOLOv8 (Python), Scene Understanding (MATLAB), and Voice Response (MATLAB)

try
    % --- 1. SETUP ---
    disp('Initializing System...');
    
    % Check Python Environment
    pe = pyenv;
    if pe.Version == ""
        % Try to locate python executable if not set
        % You might need to set it manually: pyenv('Version', 'path/to/python.exe');
        disp('Warning: Python environment not explicitly set. Using default.');
    end
    
    % Initialize Camera
    % Create webcam object. Using '1' usually defaults to the primary camera.
    % If it fails, try 'webcam(1)' or list available cams 'webcamlist'
    try
        cam = webcam(1);
    catch
        error('No webcam found. Please connect a camera.');
    end
    
    % Initialize YOLOv8 Model
    disp('Loading YOLOv8 Model (this may take a moment)...');
    try
        % Using the lightweight 'n' model for speed
        model = py.ultralytics.YOLO('yolov8n.pt');
    catch ME
        error(['Failed to load YOLOv8 model. Ensure "ultralytics" is installed in Python. Error: ' ME.message]);
    end
    
    % Initialize Text-to-Speech (SAPI for Windows)
    try
        speaker = actxserver('SAPI.SpVoice');
        speaker.Rate = 1; % Normal speed
        speaker.Volume = 100;
    catch
        warning('Text-to-Speech initialization failed. Audio feedback will be disabled.');
        speaker = [];
    end
    
    % Parameters
    confThreshold = 0.5;
    processEveryNFrames = 3; % Skip frames for performance
    frameCount = 0;
    lastSpeechTime = tic;
    speechInterval = 5; % Seconds between automatic descriptions
    
    % Create UI
    hFig = figure('Name', 'Context-Aware Image Captioning', ...
        'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none', ...
        'KeyPressFcn', @keyPressHandler);
    
    hAx = axes('Parent', hFig, 'Position', [0 0 1 1]);
    hText = uicontrol('Style', 'text', 'String', 'Initializing...', ...
        'Position', [10 10 600 30], 'FontSize', 14, ...
        'BackgroundColor', 'black', 'ForegroundColor', 'white', ...
        'HorizontalAlignment', 'left');
    
    % Flag for running loop
    global isRunning;
    isRunning = true;
    
    % Temp file for image transfer (Robust method)
    tempImgFile = fullfile(pwd, 'temp_frame.jpg');
    
    disp('System Ready. Press "q" to quit.');
    if ~isempty(speaker)
        speaker.Speak('System initialized. Starting.');
    end
    
    % --- 2. MAIN LOOP ---
    while isRunning && ishandle(hFig)
        % Acquire Image
        img = snapshot(cam);
        [h, w, ~] = size(img);
        
        frameCount = frameCount + 1;
        caption = '';
        
        if mod(frameCount, processEveryNFrames) == 0
            % Save current frame to disk for YOLO
            imwrite(img, tempImgFile);
            
            % Detect Objects (Call Python YOLO)
            % verbose=False to suppress console output
            results_list = model.predict(tempImgFile, pyargs('verbose', false));
            
            % Process first result (single image)
            result = results_list{1};
            boxes_obj = result.boxes;
            
            % Extract Data (Convert Python objects to MATLAB arrays)
            if boxes_obj.shape{1} > 0
                % xyxy boxes
                boxes_tensor = boxes_obj.xyxy.cpu().numpy();
                boxes_data = double(boxes_tensor); % [N, 4]
                
                % confidence
                conf_tensor = boxes_obj.conf.cpu().numpy();
                conf_data = double(conf_tensor); % [N, 1]
                
                % classes
                cls_tensor = boxes_obj.cls.cpu().numpy();
                cls_data = double(cls_tensor); % [N, 1]
                
                % Load class names map
                names_dict = result.names; 
                
                detectedObjects = {};
                
                % --- 3. SCENE UNDERSTANDING & DRAWING ---
                for i = 1:length(conf_data)
                    score = conf_data(i);
                    if score > confThreshold
                        box = boxes_data(i, :); % x1, y1, x2, y2
                        cls_idx = int32(cls_data(i));
                        
                        % Get Class Name
                        % Python dict keys are 0-indexed integers.
                        % result.names is a py.dict. Use .get() method.
                        label_py = names_dict.get(cls_idx); 
                        label = string(label_py);
                        
                        % --- Spatial Logic ---
                        spatial_desc = getSpatialDescription(box, w, h);
                        
                        % Store for caption
                        detectedObjects{end+1} = struct('label', label, 'spatial', spatial_desc);
                        
                        % Draw Bounding Box
                        img = insertShape(img, 'Rectangle', [box(1), box(2), box(3)-box(1), box(4)-box(2)], ...
                            'LineWidth', 3, 'Color', 'green');
                        
                        % Draw Label
                        displayText = sprintf('%s: %.2f (%s)', label, score, spatial_desc);
                        img = insertText(img, [box(1), box(2)-20], displayText, ...
                            'FontSize', 18, 'BoxColor', 'green', 'TextColor', 'black');
                    end
                end
                
                % --- 4. CAPTION GENERATION ---
                if ~isempty(detectedObjects)
                    caption = generateCaption(detectedObjects);
                else
                    caption = 'No objects detected.';
                end
                
            else
                caption = 'No objects detected.';
            end
            
            % Update UI Text
            hText.String = caption;
            
            % --- 5. VOICE RESPONSE ---
            if ~isempty(speaker) && toc(lastSpeechTime) > speechInterval && ~isempty(detectedObjects)
                % Non-blocking speech is hard in simple MATLAB script without parallel toolbox.
                % SAPI 'Speak' with flag 1 (SVSFlagsAsync) is non-blocking.
                speaker.Speak(caption, 1); 
                lastSpeechTime = tic;
            end
        end
        
        % Show Image
        imshow(img, 'Parent', hAx);
        drawnow;
    end
    
    % Cleanup
    clear cam;
    close(hFig);
    disp('System Stopped.');
    
catch ME
    disp('An error occurred:');
    disp(ME.message);
    if exist('cam', 'var')
        clear cam;
    end
end

% --- HELPER FUNCTIONS ---

function desc = getSpatialDescription(box, imgWidth, imgHeight)
    % box = [x1, y1, x2, y2]
    centerX = (box(1) + box(3)) / 2;
    % centerY = (box(2) + box(4)) / 2;
    boxArea = (box(3) - box(1)) * (box(4) - box(2));
    imgArea = imgWidth * imgHeight;
    ratio = boxArea / imgArea;
    
    % Horizontal Position
    thirdW = imgWidth / 3;
    if centerX < thirdW
        hPos = 'left';
    elseif centerX > 2 * thirdW
        hPos = 'right';
    else
        hPos = 'center';
    end
    
    % Proximity (Depth)
    if ratio > 0.15
        depth = 'near';
    elseif ratio < 0.05
        depth = 'far';
    else
        depth = 'mid-distance';
    end
    
    desc = sprintf('%s, %s', hPos, depth);
end

function finalCaption = generateCaption(objects)
    % Rule-based caption generation
    % objects is a cell array of structs
    if isempty(objects)
        finalCaption = "The scene is empty.";
        return;
    end
    
    % Create a summary sentence
    % "A person on the center, near. A cup on the left, far."
    
    % Limit to first 2-3 objects to avoid long sentences
    numObjs = min(length(objects), 3);
    sentences = strings(1, numObjs);
    
    for k = 1:numObjs
        obj = objects{k};
        sentences(k) = sprintf("A %s is on the %s", obj.label, lower(replace(obj.spatial, ',', '')));
    end
    
    finalCaption = join(sentences, '. ') + ".";
end

function keyPressHandler(~, event)
    global isRunning;
    if strcmp(event.Key, 'q')
        isRunning = false;
    end
end
