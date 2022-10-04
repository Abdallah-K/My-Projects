<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="./views/styles/discover.css"/>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.1/jquery.min.js" integrity="sha512-aVKKRRi/Q/YV+4mjoKBsE4x3H+BkegoM/em46NNlCqNTmUYADjBbeNefNxYV7giUp0VxICtqdrbqU7iVaeZNXA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css" integrity="sha512-xh6O/CkQoPOWDdYTDqeRdPCVd1SpvCA9XXcUnZS2FmJNp1coAFzvtCN9BmamE+4aHK8yyUHUSCcJHgXloTyT2A==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <link rel="shortcut icon" type="image/png" href="./views/images/logo.png"/>
    <title>Anime World</title>
</head>
<body>
    <?php include("views/components/navbar.php")?>
    <section id="DiscoverPage">
        <div class="discoversearchbar">
            <input type="text" id="searchvalue" placeholder="Enter Anime name...">
            <button id="searchbtn"><i class="fa-solid fa-magnifying-glass"></i></button>
        </div>
        <div class="discoverlists">
            <div class="discoverbox">     

            </div>
        </div>
    </section>
</body>
</html>



<script>

    $(function(){
        $("#searchbtn").click(()=>{
            var searchvalue = $("#searchvalue").val();
            var output = "";
            $.ajax({
                url:`https://api.jikan.moe/v3/search/anime?q=${searchvalue}`,
                type:"GET",
                dataType:"json",
                success:function(response){
                    $.each(response.results,(index,item)=>{
                        output +=`
                        <div class="discoverCard">
                            <div class="discoverImg">
                                <img src="${item.image_url}">
                            </div>
                            <div class="discoverinfo">
                                <div class="discoverName">
                                    <h1>${item.title}</h1>
                                </div>
                                <div class="discoverlink">
                                    <a href="./discoverCard.php?id=${item.mal_id}&search=${searchvalue}"><i class="fa-brands fa-readme"></i>Read More</a>
                                </div>
                            </div>
                        </div>`;
                    })
                    $(".discoverbox").html(output);
                }
            })
        })

    })


</script>