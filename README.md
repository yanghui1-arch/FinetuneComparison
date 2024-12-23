# FinetuneComparison
It's a repo for English Lecture to discuss four fine tune functions effectiveness. Everyone can git the code for reproduce as following ways.<br/>
```
pip install -r requirements
```
Then download the dataset of [CMDL](https://aclanthology.org/2024.findings-acl.351.pdf) which is a legal dataset and process it to be like type.
```json
{  "id": 1,
  "fact": "某地人民检察院指控：2020年5月至6月，被告人文某某、刘某某共同购买了电瓶、升压器等电捕鱼工具。2020年7月19日晚至次日凌晨，被告人文某某、姚某某、刘某某共同携带前述电瓶、升压器等工具来到重庆市渝北区悦来街道龙滩子大桥附近的后河处，在该后河水域处于禁渔区的情况下，由文某某使用工具电鱼，姚某某使用抄网接鱼，刘某某用桶装鱼，共同在后河中捕捞水产品若干，之后三名被告人与其他人将捕捞的水产品食用。2020年8月30日21时许，被告人文某某、周某某、姚某某、刘某某共谋电捕鱼后，再次携带电瓶、升压器、抄网等工具来到重庆市渝北区悦来街道龙滩子大桥附近的后河处，在该后河水域处于禁渔区的情况下，由文某某使用工具电鱼，姚某某使用抄网接鱼，周某某提桶装鱼，刘某某在附近望风，共同在后河中捕捞鲤鱼、鲫鱼、瓦氏黄颡鱼、翘嘴红鲌、鳜、赤眼鳟等水产品共9.28千克。同日，被告人刘某某、周某某、姚某某在案发现场被民警抓获，被告人文某某从现场逃离后于次日凌晨至公安机关投案。案发后，被告人文某某等使用的电鱼工具、捕捞的9.28千克渔获物均被公安机关扣押，渔获物后被销毁。针对上述指控，公诉机关举示了相应证据予以证实。",
  "defendants": ["文某某", "周某某", "姚某某", "刘某某"],
  "outcomes": [
          {"name": "文某某", "penalty": {"surveillance": 0, "detention": 5, "imprisonment": 0, "death_penalty": false, "life_imprisonment": false, "fine": 0, "fine_without_amount": false}, "charges": ["非法捕捞水产品罪"], "articles": ["340"]},
          {"name": "周某某", "penalty": {"surveillance": 0, "detention": 4, "imprisonment": 0, "death_penalty": false, "life_imprisonment": false, "fine": 0, "fine_without_amount": false}, "charges": ["非法捕捞水产品罪"], "articles": ["340"]},
          {"name": "姚某某", "penalty": {"surveillance": 0, "detention": 5, "imprisonment": 0, "death_penalty": false, "life_imprisonment": false, "fine": 0, "fine_without_amount": false}, "charges": ["非法捕捞水产品罪"], "articles": ["340"]},
          {"name": "刘某某", "penalty": {"surveillance": 0, "detention": 5, "imprisonment": 0, "death_penalty": false, "life_imprisonment": false, "fine": 0, "fine_without_amount": false}, "charges": ["非法捕捞水产品罪"], "articles": ["340"]}
  ]
}
```
And then just update the train settings hyper-parameters~ the code will run!!!
