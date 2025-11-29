from tokenization_mlm import MLMTokenizer
from transformers import MBartForConditionalGeneration

# For DRS-to-text generation
tokenizer = MLMTokenizer.from_pretrained('laihuiyuan/DRS-LMM', src_lang='<drs>')
model = MBartForConditionalGeneration.from_pretrained('laihuiyuan/DRS-LMM')

# Gold text: The court is adjourned until 3:00 p.m. on March 1st.
# inp_ids = tokenizer.encode(
#     "court.n.01 time.n.08 EQU now adjourn.v.01 Theme -2 Time -1 Finish +1 time.n.08 ClockTime 15:00 MonthOfYear 3 DayOfMonth 1",
#     return_tensors="pt")
inp_ids = tokenizer.encode(
    "In order to test the system, we feed it this sentence.",
    return_tensors="pt")

foced_ids = tokenizer.encode("en_XX", add_special_tokens=False, return_tensors="pt")
outs = model.generate(input_ids=inp_ids, forced_bos_token_id=foced_ids.item(), num_beams=5, max_length=150)
text = tokenizer.decode(outs[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)