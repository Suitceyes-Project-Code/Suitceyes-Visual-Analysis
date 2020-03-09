<?php
    require('access.php');
?>
<?php
	function check_file_uploaded_name ($filename) {
    	return (bool) ((preg_match("`^[-0-9A-Z_\.]+$`i",$filename)) ? true : false);
	}

	function check_file_uploaded_length ($filename) {
   		return (bool) ((mb_strlen($filename,"UTF-8") < 225) ? true : false);
	}

    function check_file_size ($file_size) {
        return (bool) ( ($file_size < 20000000) ? true : false);
    }

	if(!empty($_FILES)) {
		$temp_name = $_FILES["file"]["tmp_name"];
		$new_name = basename($_FILES["file"]["name"]);
        $file_size = $_FILES["file"]["size"];
        #The directory where the files are uploaded
		$upload_dir = "";
		
		if(preg_match("/\.+(jpg|JPG|jpeg|JPEG|png|PNG|json|txt)$/", $new_name) and check_file_uploaded_name($new_name) and check_file_uploaded_length($new_name) and check_file_size($file_size)) {
			move_uploaded_file($temp_name, "$upload_dir/$new_name");
			echo "UPLOAD OK";
		}
		else {
			echo "UPLOAD FAILED. ERROR IN FILE NAME OR SIZE";
		}
	}
	else {
		#echo"NO FILES TO UPLOAD";
        require('404.html');
	}
?>
