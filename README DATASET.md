---
license: cc-by-sa-3.0
---
## Structure

The set of base document types for MIDV-2020 comprises 10 document types, each 
present in previously published MIDV-500 and MIDV-2019 datasets. The identity 
document types of MIDV-2020 are listed in Table DOCTYPES. 100 sample documents 
were created for each of the 10 document types present in the dataset.

| #  | Code                 | Description                 | MIDV-500 code |
|----|----------------------|-----------------------------|---------------|
| 1  | alb_id               | ID Card of Albania          | 01            |
| 2  | aze_passport         | Passport of Azerbaijan      | 05            |
| 3  | esp_id               | ID Card of Spain            | 21            |
| 4  | est_id               | ID Card of Estonia          | 22            |
| 5  | fin_id               | ID Card of Finland          | 24            |
| 6  | grc_passport         | Passport of Greece          | 25            |
| 7  | lva_passport         | Passport of Latvia          | 32            |
| 8  | rus_internalpassport | Internal passport of Russia | 39            |
| 9  | srb_passport         | Passport of Serbia          | 41            |
| 10 | svk_id               | ID Card of Slovakia         | 42            |

Original template images are placed in the `templates.tar` archive:
templates.tar: /images/ /CODE/ 00.jpg01.jpg... 99.jpg/annotations/CODE.json

Upright scans are placed in the `scan_upright.tar` archive. 

scan_upright.tar: /images/ /CODE/ 00.jpg01.jpg... 99.jpg/annotations/ CODE.json


Rotated scans are placed in the same manner in `scan_rotated.tar`
scan_rotated.tar: /images/ /CODE/ 00.jpg01.jpg... 99.jpg/annotations/CODE.json

Photos are placed in the same manner in `photo.tar`:
photo.tar: /images/ /CODE/ 00.jpg01.jpg... 99.jpg/annotations/CODE.json

Photos were captured with different conditions, the condition can be identified
by the photo number:

| Capturing conditions and smartphone models | Samsung S10 | Apple iPhone XR |
|--------------------------------------------|-------------|-----------------|
| Low lighting                               | 80-89       | 70-79           |
| Keyboard in the background                 | 35-39       | 30-34           |
| Natural lighting, outdoors                 | 45-49       | 40-44           |
| Table in the background                    | 55-59       | 50-54           |
| Cloth in the background                    | 95-99       | 90-94           |
| Text documents in the background           | 25-29       | 20-24           |
| Projective distortions                     | 10-19       | 00-09           |
| Highlight present                          | 65-69       | 60-64           |


All annotations are made using VGG Image Annotator (VIA) v2.0.11, which can be
obtained via this [link](https://www.robots.ox.ac.uk/~vgg/software/via/downloads/via-2.0.11.zip).

The developer's website: [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/)

--------------------------------------------------------------------------------