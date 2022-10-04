<?php


include_once "./db.php";

class PostModel{


    public function __construct()
    {
        $response = [];
        $action = $_REQUEST['action'];
        if($action == "select"){
            $response = self::SelectData($_POST);
        }
        if($action == "selectcon"){
            $response = self::SelectCon($_POST);
        }

        echo json_encode($response);
    }



    public static function SelectData($tablename){
        $tablename = $tablename['tablename'];
        $sql = "SELECT * FROM $tablename";
        $db = new DB();
        return $db->execselect($sql);
    }

    public static function SelectCon($Condition){
        $id = $Condition['id'];
        $table = $Condition['tablename'];
        $sql = "SELECT * FROM $table WHERE id='$id'";
        $db = new DB();
        return $db->execselect($sql);
    }
}



if(isset($_REQUEST['action'])){
    new PostModel();
}





?>