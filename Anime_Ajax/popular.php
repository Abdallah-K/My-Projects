<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="./views/styles/popular.css"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css" integrity="sha512-xh6O/CkQoPOWDdYTDqeRdPCVd1SpvCA9XXcUnZS2FmJNp1coAFzvtCN9BmamE+4aHK8yyUHUSCcJHgXloTyT2A==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.1/jquery.min.js" integrity="sha512-aVKKRRi/Q/YV+4mjoKBsE4x3H+BkegoM/em46NNlCqNTmUYADjBbeNefNxYV7giUp0VxICtqdrbqU7iVaeZNXA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <link rel="shortcut icon" type="image/png" href="./views/images/logo.png"/>
    <title>Anime World</title>
</head>
<body>
    <?php include("views/components/navbar.php")?>
    <section id="PopularPage">
        <div class="popularbox">
            <div class="populartitle">
                <h1>Popular Anime</h1>
            </div>
            <div class="popular-lists">


            </div>
        </div>
    </section>

    
</body>
</html>



<script>

    $(function(){

        var popularout = "";
        $.post("API/PostModel.php",{
            action:"select",
            tablename:"populardata",
        },function(data){
            data = $.parseJSON(data);
            $.each(data,(index,popitem)=>{
                popularout +=`
                    <div class="popularCard">
                        <div class="popularinfo">
                            <div class="poptitle">
                                <h1>${popitem.name}</h1>
                            </div>
                            <div class="popbdy">
                                <p>${popitem.description}</p>
                            </div>
                        </div>
                        <div class="popularimg">
                            <img src="./views/images/popular/${popitem.image}">
                        </div>
                    </div>`;
            })
            $(".popular-lists").html(popularout);
        })


    })

</script>