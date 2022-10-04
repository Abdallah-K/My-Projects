<?php

class DB{

    private $host = "localhost";
    private $user = "root";
    private $pwd = "";
    private $dbName = "anime";

    public function connect(){
        $con = mysqli_connect($this->host,$this->user,$this->pwd,$this->dbName);
        return $con;
    }

    public static function escape($user_value){
        $db = new DB();
        $con = $db->connect();
        return mysqli_real_escape_string($con,$user_value);
    }

    public function execselect($query){
        $db = new DB();
        $con = $db->connect(); 
        $result = mysqli_query($con,$query);
        $data_array = [];
        while($rows = mysqli_fetch_assoc($result)){
            $data_array[] = $rows;
        }
        return $data_array;
    }


    public function execinsertdel($query){
        $response =[
            "status" =>"Error",
            "message" => "Error,try again"
        ];
        $db = new DB();
        $con = $db->connect();
        $result = mysqli_query($con,$query);
        if($result){
            $response["status"] = "success";
            $response["message"] = "Record Inserted";
        }
        return $response;
    }

 

}



?>