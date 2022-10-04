<?php

$id = $_GET['id'];

if(!isset($id)){
    header("location:./index.php");
}


?>


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="views/styles/imdbcard.css"/>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.1/jquery.min.js" integrity="sha512-aVKKRRi/Q/YV+4mjoKBsE4x3H+BkegoM/em46NNlCqNTmUYADjBbeNefNxYV7giUp0VxICtqdrbqU7iVaeZNXA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <link rel="shortcut icon" type="image/png" href="./views/images/logo.png"/>
    <title>Anime World</title>
</head>
<body>
    <input type="hidden" id="idVal" name="id_data" value="<?=$id?>">
    <?php include("views/components/navbar.php")?>
    <section id="ImdbCard">
        <div class="imdbback">
            <a href="./home.php#IMDBPage">Back</a>
        </div>
        <div class="imdbcardlist">

        </div>

    </section>
    


</body>
</html>



<script>

    $(function(){

        var input = $("#idVal").val();
        var IMDBCARD = "";


        $.post("API/PostModel.php",{
            action:"selectcon",
            tablename:"imdbdata",
            id:input,
        },function(data){
            data = $.parseJSON(data);
            IMDBCARD +=`
                <div class="imdcard">   
                    <div class="imdbimg">
                        <img src="./views/images/IMDB/${data[0].image}">
                    </div>
                    <div class="imdbinfo">
                        <div class="imdbtitle">
                            <h1>${data[0].name}</h1>
                            <h2>${data[0].rate}</h2>
                        </div>
                        <div class="imdbdes">
                            <p>${data[0].description}</p>
                        </div>
                    </div>
                </div>`;
            $(".imdbcardlist").html(IMDBCARD);
        })

    
    })


</script>