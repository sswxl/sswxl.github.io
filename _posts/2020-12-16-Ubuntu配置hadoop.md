---
layout: post
title: Ubuntu20.04配置eclipse并创建HDFS项目
mathjax: true
tags: Ubuntu hadoop eclipse
author: Wxl
date: 2020-12-16
header-style: text
excerpt_separator: <!--more-->
---

1.Ubuntu配置Java和Hadoop

参考林子雨教程http://dblab.xmu.edu.cn/blog/285/

2.配置eclipse

[Hadoop3.1.3安装教程伪分布式配置_Hadoop3.1.3/Ubuntu20.04](http://dblab.xmu.edu.cn/blog/2441-2/) 

[mapreduce](https://www.bilibili.com/video/BV1JT4y1g7nM?p=83)

根本不需要hadoop用户

1.Ubuntu配置Java和Hadoop

参考林子雨教程http://dblab.xmu.edu.cn/blog/285/



\3. 打开eclipse创建工程

参考林子雨教程http://dblab.xmu.edu.cn/blog/290-2/

在导包时，除了链接中要导入的包外，还要导入**hadoop-hdfs-client-3.1.1.jar，**否则会报错。

*如果忘了导入该包*

*可以右键点击Java项目名，依次点击“Build path”,“Configue Build Path”,"Libraries"。*

\4. 如果建eclipse工程卡住

打开终端，依次输入

​    export SWT_GTK3=0

​    cd /usr/lib/eclipse （即eclipse.ini的路径）

​    sudo vi eclipse.ini

然后在--launcher.appendVmargs前面添加两行

​    --launcher.GTK_version

​    2

 [Linux环境变量配置全攻略](https://www.cnblogs.com/youyoui/p/10680329.html)

```shell
export HADOOP_HOME=/usr/local/hadoop
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH
```

Shell

保存后，务必执行 `source ~/.bashrc` 使变量设置生效，然后再次执行 `./sbin/start-dfs.sh` 启动 Hadoop。

到此，本地库就能正常加载了。可以通过以下命令进行自检：

```
  hadoop checknative –a 
```

export HADOOP_HOME=本机的hadoop安装路径
export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH

保存后退出。