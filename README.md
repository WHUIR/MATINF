# MATINF - Multitask Chinese NLP Dataset
The dataset and PyTorch Implementation for ACL 2020 paper ["MATINF: A Jointly Labeled Large-Scale Dataset for Classification, Question Answering and Summarization"](https://arxiv.org/abs/2004.12302).

## Citation
If you use the dataset or code in your research, please kindly cite our work:

```bibtex
@inproceedings{xu-etal-2020-matinf,
    title = "{MATINF}: A Jointly Labeled Large-Scale Dataset for Classification, Question Answering and Summarization",
    author = "Xu, Canwen  and
      Pei, Jiaxin  and
      Wu, Hongtao  and
      Liu, Yiyu  and
      Li, Chenliang",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.330",
    pages = "3586--3596",
}
```

## Dataset
You can get MATINF dataset by signing [the agreement on Google Form](https://forms.gle/nkH4LVE4iNQeDzsc9) to request the access. You will get the download link and the zip password after filling the form. Please use [7zip](https://www.7-zip.org/) to decompress.

**ALL USE MUST BE NON-COMMERCIAL!!**

## Code
Please manually change the `stage` variable in `main()` to toggle from different training phases.

Then run:
```bash
python run.py
```
Code credit: [Hongtao Wu](mailto:wuhongtao@whu.edu.cn?cc=xucanwen@whu.edu.cn)
