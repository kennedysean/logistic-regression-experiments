data = importdata('data.txt');
labels = importdata('labels.txt');
combined_data = [data labels];
writematrix(combined_data(1:2000,1:58), 'training.txt');
writematrix(combined_data(2001:4601,1:58), 'test.txt');