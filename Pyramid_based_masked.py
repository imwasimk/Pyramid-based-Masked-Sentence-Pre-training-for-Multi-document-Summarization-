{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "36735b40",
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import (\n",
    "    AutoTokenizer,\n",
    "    LEDForConditionalGeneration,\n",
    ")\n",
    "from datasets import load_dataset, load_metric\n",
    "import torch"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b9f59338",
   "metadata": {},
   "source": [
    
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5faeeffc",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using custom data configuration default\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3d83cdf56f2a4ee0bea672c1f8f5264d",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Downloading:   0%|          | 0.00/257M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "0 examples [00:00, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "0 examples [00:00, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "0 examples [00:00, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1daf64cd6b7c4556b9d695ae9d6add33",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/3 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "dataset=load_dataset('multi_news')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a9137321",
   "metadata": {},
   "source": [
    
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "890f434e",
   "metadata": {},
   "outputs": [],
   "source": [
    "TOKENIZER = AutoTokenizer.from_pretrained(PRIMER_path)\n",
    "MODEL = LEDForConditionalGeneration.from_pretrained(PRIMER_path)\n",
    "MODEL.gradient_checkpointing_enable()\n",
    "PAD_TOKEN_ID = TOKENIZER.pad_token_id\n",
    "DOCSEP_TOKEN_ID = TOKENIZER.convert_tokens_to_ids(\"<doc-sep>\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bd5aabfc",
   "metadata": {},
   "source": [
    "We then define the functions to pre-process the data, as well as the function to generate summaries."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "7bfecd47",
   "metadata": {},
   "outputs": [],
   "source": [
    "def process_document(documents):\n",
    "    input_ids_all=[]\n",
    "    for data in documents:\n",
    "        all_docs = data.split(\"|||||\")[:-1]\n",
    "        for i, doc in enumerate(all_docs):\n",
    "            doc = doc.replace(\"\\n\", \" \")\n",
    "            doc = \" \".join(doc.split())\n",
    "            all_docs[i] = doc\n",
    "\n",
    "        #### concat with global attention on doc-sep\n",
    "        input_ids = []\n",
    "        for doc in all_docs:\n",
    "            input_ids.extend(\n",
    "                TOKENIZER.encode(\n",
    "                    doc,\n",
    "                    truncation=True,\n",
    "                    max_length=4096 // len(all_docs),\n",
    "                )[1:-1]\n",
    "            )\n",
    "            input_ids.append(DOCSEP_TOKEN_ID)\n",
    "        input_ids = (\n",
    "            [TOKENIZER.bos_token_id]\n",
    "            + input_ids\n",
    "            + [TOKENIZER.eos_token_id]\n",
    "        )\n",
    "        input_ids_all.append(torch.tensor(input_ids))\n",
    "    input_ids = torch.nn.utils.rnn.pad_sequence(\n",
    "        input_ids_all, batch_first=True, padding_value=PAD_TOKEN_ID\n",
    "    )\n",
    "    return input_ids\n",
    "\n",
    "\n",
    "def batch_process(batch):\n",
    "    input_ids=process_document(batch['document'])\n",
    "    # get the input ids and attention masks together\n",
    "    global_attention_mask = torch.zeros_like(input_ids).to(input_ids.device)\n",
    "    # put global attention on <s> token\n",
    "\n",
    "    global_attention_mask[:, 0] = 1\n",
    "    global_attention_mask[input_ids == DOCSEP_TOKEN_ID] = 1\n",
    "    generated_ids = MODEL.generate(\n",
    "        input_ids=input_ids,\n",
    "        global_attention_mask=global_attention_mask,\n",
    "        use_cache=True,\n",
    "        max_length=1024,\n",
    "        num_beams=5,\n",
    "    )\n",
    "    generated_str = TOKENIZER.batch_decode(\n",
    "            generated_ids.tolist(), skip_special_tokens=True\n",
    "        )\n",
    "    result={}\n",
    "    result['generated_summaries'] = generated_str\n",
    "    result['gt_summaries']=batch['summary']\n",
    "    return result"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "250f3053",
   "metadata": {},
   "source": [
    "Next, we simply run the model on 10 data examples (or any number of examples you want)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "631ead96",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "282e6e400d8745859d31eb1e7b9d28aa",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/5 [00:00<?, ?ba/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import random\n",
    "data_idx = random.choices(range(len(dataset['test'])),k=10)\n",
    "dataset_small = dataset['test'].select(data_idx)\n",
    "result_small = dataset_small.map(batch_process, batched=True, batch_size=2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab2dd96d",
   "metadata": {},
   "source": [
    "After getting all the results, we load the evaluation metric. \n",
    "\n",
    "\n",
    "(Note in the original code, we didn't use the default aggregators, instead, we simply take average over all the scores.\n",
    "We simply use 'mid' in this notebook)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "d8f31905",
   "metadata": {},
   "outputs": [],
   "source": [
    "rouge = load_metric(\"rouge\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "81814601",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
      
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result_small['generated_summaries']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "8d9923d2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Score(precision=0.509437078378281, recall=0.43832461548851936, fmeasure=0.4644188580686355)\n",
      "Score(precision=0.17689604682544763, recall=0.14564519595131636, fmeasure=0.1581222605371442)\n",
      "Score(precision=0.2362355904256852, recall=0.19669444890277293, fmeasure=0.21194685290367665)\n"
     ]
    }
   ],
   "source": [
    "score=rouge.compute(predictions=result_small[\"generated_summaries\"], references=result_small[\"gt_summaries\"])\n",
    "print(score['rouge1'].mid)\n",
    "print(score['rouge2'].mid)\n",
    "print(score['rougeL'].mid)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "6e5cdd3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "c2c76256",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[4496, 1390, 2088, 2130, 1604]"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "random.choices(range(5000),k=5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7e763cc4",
   "metadata": {},
   "source": [
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fc3708f8",
   "metadata": {},
   "source": [
  
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
