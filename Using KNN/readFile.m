% Description: This function calls files from different locations. It
% returns output in structure format.

function [out info] = readFile(search_item,set_folder_location)
    find_location_and_item = dir(fullfile(set_folder_location,search_item));
    total_item_found = length(find_location_and_item);
    
    if total_item_found == 0;
        disp('Nothing found')
        info=[];
        out = [];
    else
        for i = 1:total_item_found
            a{i} = load(fullfile(set_folder_location,find_location_and_item(i).name));
            r = size(a{i},1);       % row size
            sr = num2str(r);        % convert to string
            c = size(a{i},2);       % column size
            cr = num2str(c);        % convert to string
            rc = ['file name: ', find_location_and_item(i).name, '     ', sr, 'x', cr];
            info(i,:) = {rc};
        end
        out = a;
    end      
end

%% --------------------------- Example:--------------------------- %%
% clc; clear all; close all;
% search_item = '*reg_unreg_BDCT*.txt';
% set_folder_location = 'D:\academic\LU\project\Palm print detection\Code by Us\ROC';
% [out info] = readFile(search_item,set_folder_location);