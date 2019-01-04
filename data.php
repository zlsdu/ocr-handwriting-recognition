<?php 
//header("content-type:text/html;charset=utf8");
$city = empty($_GET['city']) ? "上海" : $_GET['city'];
echo file_get_contents("http://wthrcdn.etouch.cn/weather_mini?city={$city}");
 ?>