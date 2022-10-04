<?php

$id = $_GET['id'];
$search = $_GET['search'];

if(!isset($id)){
    header("location:./discover.php");
}


?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="./views/styles/discovercard.css"/>    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.1/jquery.min.js" integrity="sha512-aVKKRRi/Q/YV+4mjoKBsE4x3H+BkegoM/em46NNlCqNTmUYADjBbeNefNxYV7giUp0VxICtqdrbqU7iVaeZNXA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css" integrity="sha512-xh6O/CkQoPOWDdYTDqeRdPCVd1SpvCA9XXcUnZS2FmJNp1coAFzvtCN9BmamE+4aHK8yyUHUSCcJHgXloTyT2A==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <link rel="shortcut icon" type="image/png" href="./views/images/logo.png"/>
    <title>Anime World</title>
</head>
<body>
    <?php include("views/components/navbar.php")?>
    <section id="DiscPage">
        <div class="discback">
            <a href="./discover.php">Back</a>
        </div>
        <div class="disccardsec">

        </div>


    </section>
    
</body>
</html>



<script>

    $(function(){
        var newcard = "";
        $.ajax({
            type:"GET",
            url: `https://api.jikan.moe/v3/search/anime?q=${"<?php echo $search?>"}`,
            datatype:"json",
            success: function(response){
                jQuery.grep(response.results,function(animeinfo){ 
                    if(animeinfo.mal_id == "<?php echo $id?>"){
                        newcard+=`
                        <div class="discCard">
                            <div class="discImg">
                                <img src="${animeinfo['image_url']}">
                            </div>
                            <div class="discInfo">

                                <div class="discone">
                                    <h1>${animeinfo['title']}</h1>
                                </div>
                                <div class="disctwo">
                                    <p>${animeinfo['synopsis']}</p>
                                </div>
                                <div class="discthree">
                                    <div class="type">
                                        <h2>${animeinfo['type']}</h2>
                                    </div>
                                    <div class="epi">
                                        <h2>${animeinfo['episodes']}</h2>
                                    </div>
                                    <div class="score">
                                        <h2>${animeinfo['score']}</h2>
                                    </div>
                                </div>
                                <div class="discfour">
                                        <h3>Start-data: ${animeinfo['start_date']}</h3>
                                        <h3>End-data: ${animeinfo['end_date']}</h3>
                                </div>
                                <div class="discfive">
                                    <a target="_blank" href="${animeinfo['url']}">See More</a>
                                </div>

                            </div>
                        </div>  
                        `;
                    $(".disccardsec").html(newcard);

                    }
                });
            }
        });
    })




</script>