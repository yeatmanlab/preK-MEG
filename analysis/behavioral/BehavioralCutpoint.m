%% Divide sample into high and low letter knowledge

d = readtable('RDRPRepository-PREK_DATA_LABELS_2020-08-23_1500.csv');
d = d(strcmp(d.WhatStudyIsTheSubjectBeingEnrolledIn_ThisResponseWillBeUsedToSe,'prek_pre_language' ) |...
    strcmp(d.WhatStudyIsTheSubjectBeingEnrolledIn_ThisResponseWillBeUsedToSe,'prek_pre_letter' ),:);
d = d(~isnan(d.Pre_ReaderLetterKnowledge_OfLetters),:);
cutpointZ = 0;% This was the trough in the bimodal distribution
t = table(d.RecordID, d.Pre_ReaderLetterKnowledge_OfLetters, d.Pre_ReaderLetterKnowledge_OfLetters_Lowercase_,...
    d.Pre_ReaderLetterKnowledge_OfSounds,d.Pre_ReaderLetterKnowledge_OfSounds_Lowercase_,'VariableNames',...
    {'subID' 'UpperName' 'LowerName' 'UpperSound' 'LowerSound'});
t.LeterKnowledge = mean(zscore(table2array(t(:,2:5))),2);
t.HighLow = zeros(size(t,1),1);
%t.HighLow(t.LeterKnowledge > nanmedian(t.LeterKnowledge))=1;
t.HighLow(t.LeterKnowledge > cutpointZ)=1;
figure;
subplot(2,2,1);hold
plot(t.UpperName(t.HighLow==1),t.UpperSound(t.HighLow==1),'ro')
plot(t.UpperName(t.HighLow==0),t.UpperSound(t.HighLow==0),'bo')
xlabel('Name');ylabel('Sound');title('Upper')
subplot(2,2,2);hold
plot(t.LowerName(t.HighLow==1),t.LowerSound(t.HighLow==1),'ro');
plot(t.LowerName(t.HighLow==0),t.LowerSound(t.HighLow==0),'bo')
xlabel('Name');ylabel('Sound');title('Lower')

subplot(2,2,3);hold
plot(t.LeterKnowledge(t.HighLow==1),t.UpperName(t.HighLow==1)+t.UpperSound(t.HighLow==1),'ro')
plot(t.LeterKnowledge(t.HighLow==0),t.UpperName(t.HighLow==0)+t.UpperSound(t.HighLow==0),'bo')
xlabel('Letter Knowledge');ylabel('Upper Sum');title('Upper')


subplot(2,2,4);hold
plot(t.LeterKnowledge(t.HighLow==1),t.LowerName(t.HighLow==1)+t.LowerSound(t.HighLow==1),'ro')
plot(t.LeterKnowledge(t.HighLow==0),t.LowerName(t.HighLow==0)+t.LowerSound(t.HighLow==0),'bo')
xlabel('Letter Knowledge');ylabel('Lower Sum');title('Lower')

figure;subplot(2,1,1);hold
[f, xi] = ksdensity(t.LeterKnowledge);
plot(xi,f,'-k');plot([cutpointZ; cutpointZ],[min(f); max(f)],'-r');

subplot(2,1,2);hold
h=histogram(t.LeterKnowledge,xi);
plot([cutpointZ; cutpointZ],[0; max(h.Values)],'-r');
%% save table
writetable(t,'LetterKnowledge.csv')

