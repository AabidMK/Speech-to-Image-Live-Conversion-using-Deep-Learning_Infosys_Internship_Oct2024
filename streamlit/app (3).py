
import os
import pandas as pd # type: ignore
from datasets import load_dataset, Audio # type: ignore
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer # type: ignore
import torch # type: ignore
from torch.nn.utils.rnn import pad_sequence # type: ignore

csv_path = r'C:\Users\Sabarinathan S\Desktop\streamlit\Dataset_1\Dataset_1\Recordings\audio__details.csv'
audio_folder_path = r'C:\Users\Sabarinathan S\Desktop\streamlit\Dataset_1\Dataset_1\Recordings\Train'
df = pd.read_csv(csv_path)
df['File_name'] = df['File_name'].apply(lambda x: os.path.abspath(os.path.join(audio_folder_path, os.path.basename(x))))
df.to_csv(csv_path, index=False)


dataset = load_dataset('csv', data_files=csv_path)

dataset = dataset.cast_column('File_name', Audio(sampling_rate=16000)).rename_column('File_name', 'audio').rename_column('phrase', 'sentence')

print(dataset['train'][0]['audio'])

# Initialize Whisper processor
processor = WhisperProcessor.from_pretrained("openai/whisper-base")

def prepare_dataset(batch):
    # Process audio to input features
    batch["input_features"] = processor(batch["audio"]["array"], sampling_rate=16000).input_features[0]
    # Process transcription to labels
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

# Remove unnecessary columns from the dataset
# available_columns = set(dataset.column_names)
# columns_to_remove = [
#     'audio_clipping', 'audio_clipping:confidence', 'background_noise_audible', 
#     'background_noise_audible:confidence', 'overall_quality_of_the_audio', 
#     'quiet_speaker', 'quiet_speaker:confidence', 'speaker_id', 'file_download', 
#     'prompt', 'writer_id','Set'
# ]
# existing_columns_to_remove = [col for col in columns_to_remove if col in available_columns]

# Apply processing function and remove existing columns
dataset = dataset.map(prepare_dataset)

# Load Whisper model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

training_args = Seq2SeqTrainingArguments(
    output_dir="whisper-finetuned",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=3,
    fp16=False,
    save_steps=100,
    logging_steps=10,
    evaluation_strategy="no",
    save_total_limit=2,
)

# Define custom data collator
class DataCollatorForWhisper:
    def __call__(self, features):  
        input_features = [torch.tensor(feature["input_features"]) for feature in features]
        labels = [torch.tensor(feature["labels"]) for feature in features]
        input_features_padded = pad_sequence(input_features, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_features": input_features_padded,
            "labels": labels_padded
        }

# Initialize data collator
data_collator = DataCollatorForWhisper()

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
)

# Train model
trainer.train()

# Save model and processor in both .safetensors and .bin formats (disable safe serialization for .bin)
model.save_pretrained("whisper-finetuned", safe_serialization=False)
processor.save_pretrained("whisper-finetuned")

# Additionally save model weights in .bin format
torch.save(model.state_dict(), "whisper-finetuned/pytorch_model.bin")