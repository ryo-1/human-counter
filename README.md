# human-counter
画像保存用リポジトリ <br>

画像リポジトリの作成方法 <br>
① 現在のプロジェクトとは別の箇所にディレクトリ作成 <br>
② 元のプロジェクトの.gitを作成したディレクトリへコピー <br>
  ex. 元のプロジェクトの.gitがある場所で下記コマンド実行 第２引数はコピー先 <br>
  cp -R .git /Users/iwama/Desktop/python/画像 <br>
③ 作成したディレクトリでブランチ作成＆移動 <br>
  git checkout -b 画像 origin/画像 <br>
④ remoteからpull  <br>
  git pull origin 画像 <br>
⑤ 画像追加して add して commit して origin 画像 へ push <br>
