<?php
require "./Upload.class.php";

$upload = new Upload();


$upload->maxSize   =     3145728000 ;// 设置附件上传大小
$upload->exts      =     array('jpg', 'gif', 'png', 'jpeg');// 设置附件上传类型
$upload->rootPath  =     './uploads/'; // 设置附件上传根目录
//$upload->autoSub  =    "";  //子目录

$info = $upload->upload();

echo "aaaa";

//定义结果
$result = array();
if (!$info) {

	$result["error"] = 1;
	$result["message"] = $upload->getError();

} else {
	$result["error"] = 0;
	$result["data"] = $info;
}


echo json_encode($result);



